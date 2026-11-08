from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .selective_momentum_memory import RealMamba


PHASE_NAMES = ("stationary", "accelerating", "cruising", "decelerating")


def _assert_shape(tensor: torch.Tensor, expected, name: str) -> None:
    actual = tuple(tensor.shape)
    if tuple(expected) != actual:
        raise ValueError(f"{name} shape mismatch: expected {tuple(expected)}, got {actual}")


def cumulative_to_increments(cumulative: torch.Tensor) -> torch.Tensor:
    if cumulative.shape[-1] != 2:
        raise ValueError(f"Expected trajectory xy dimension 2, got {tuple(cumulative.shape)}")
    first = cumulative[..., :1, :]
    rest = cumulative[..., 1:, :] - cumulative[..., :-1, :]
    return torch.cat([first, rest], dim=-2)


def trajectory_proxy_features(
    cumulative_plan: torch.Tensor,
    dt: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return token geometry, longitudinal speed, and acceleration proxies.

    Coordinates follow the audited MomAD convention: x is lateral and +y is
    ego-forward longitudinal progress.
    """
    if cumulative_plan.dim() != 5 or cumulative_plan.shape[-2:] != (12, 2):
        raise ValueError(
            "Expected cumulative_plan [B,3,6,12,2], got "
            f"{tuple(cumulative_plan.shape)}"
        )
    delta = cumulative_to_increments(cumulative_plan)
    y = cumulative_plan[..., 1:2]
    x = cumulative_plan[..., 0:1]
    vy = delta[..., 1:2] / float(dt)
    ay = torch.zeros_like(vy)
    ay[..., 1:, :] = (vy[..., 1:, :] - vy[..., :-1, :]) / float(dt)
    geom = torch.cat([cumulative_plan, delta, y, x, vy, ay], dim=-1)
    return geom, vy, ay


def build_phase_labels(
    gt_ego_fut_trajs: torch.Tensor,
    gt_ego_fut_masks: torch.Tensor,
    dt: float = 0.5,
    v_stop: float = 0.5,
    a_thr: float = 0.5,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build horizon-wise phase labels for supervision only.

    This function is intentionally separate from module forward. It may be used
    by losses or offline annotation statistics, never by token construction.
    """
    if gt_ego_fut_trajs.dim() != 3 or gt_ego_fut_trajs.shape[-2:] != (12, 2):
        raise ValueError(
            "Expected gt_ego_fut_trajs [B,12,2], got "
            f"{tuple(gt_ego_fut_trajs.shape)}"
        )
    if gt_ego_fut_masks.shape != gt_ego_fut_trajs.shape[:2]:
        raise ValueError(
            "gt_ego_fut_masks shape mismatch: "
            f"{tuple(gt_ego_fut_masks.shape)} vs {tuple(gt_ego_fut_trajs.shape[:2])}"
        )
    vy = gt_ego_fut_trajs[..., 1] / float(dt)
    ay = torch.zeros_like(vy)
    ay[:, 1:] = (vy[:, 1:] - vy[:, :-1]) / float(dt)
    valid = gt_ego_fut_masks.to(dtype=torch.bool)
    labels = torch.full_like(vy, fill_value=2, dtype=torch.long)
    stationary = vy.abs() < float(v_stop)
    accelerating = (ay > float(a_thr)) & ~stationary
    decelerating = (ay < -float(a_thr)) & ~stationary
    labels[stationary] = 0
    labels[accelerating] = 1
    labels[decelerating] = 3
    labels[~valid] = -1
    return labels, valid


def phase_supervision_loss(
    phase_logits: torch.Tensor,
    gt_ego_fut_trajs: torch.Tensor,
    gt_ego_fut_masks: torch.Tensor,
    dt: float = 0.5,
    v_stop: float = 0.5,
    a_thr: float = 0.5,
) -> Tuple[torch.Tensor, Dict[str, object]]:
    labels, valid = build_phase_labels(
        gt_ego_fut_trajs,
        gt_ego_fut_masks,
        dt=dt,
        v_stop=v_stop,
        a_thr=a_thr,
    )
    if phase_logits.shape[:2] != labels.shape or phase_logits.shape[-1] != len(PHASE_NAMES):
        raise ValueError(
            "phase_logits must be [B,12,4], got "
            f"{tuple(phase_logits.shape)} for labels {tuple(labels.shape)}"
        )
    flat_valid = valid.reshape(-1)
    valid_count = int(flat_valid.sum().detach().cpu().item())
    total_count = int(valid.numel())
    if valid_count == 0:
        zero = phase_logits.sum() * 0.0
        return zero, {
            "valid_count": 0,
            "total_count": total_count,
            "valid_label_coverage": 0.0,
            "phase_counts": {name: 0 for name in PHASE_NAMES},
        }
    loss = F.cross_entropy(
        phase_logits.reshape(-1, phase_logits.shape[-1])[flat_valid],
        labels.reshape(-1)[flat_valid],
        reduction="mean",
    )
    label_flat = labels.reshape(-1)[flat_valid]
    counts = torch.bincount(label_flat, minlength=len(PHASE_NAMES)).detach().cpu()
    stats = {
        "valid_count": valid_count,
        "total_count": total_count,
        "valid_label_coverage": float(valid_count / max(total_count, 1)),
        "phase_counts": {
            name: int(counts[idx].item()) for idx, name in enumerate(PHASE_NAMES)
        },
        "phase_ratios": {
            name: float(counts[idx].item() / max(valid_count, 1))
            for idx, name in enumerate(PHASE_NAMES)
        },
    }
    return loss, stats


class LongitudinalPhaseHorizonMamba(nn.Module):
    """Candidate-isolated horizon refiner for MomAD planning candidates.

    Forward accepts only causal model state and coarse predictions. GT phase,
    GT trajectory, and GT command are intentionally absent from the signature.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        token_dims: int = 256,
        phase_embed_dims: int = 32,
        horizon_steps: int = 12,
        num_commands: int = 3,
        num_modes: int = 6,
        num_phase: int = 4,
        dt: float = 0.5,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        require_real_mamba: bool = True,
        allow_mamba_fallback: bool = False,
    ):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.token_dims = int(token_dims)
        self.phase_embed_dims = int(phase_embed_dims)
        self.horizon_steps = int(horizon_steps)
        self.num_commands = int(num_commands)
        self.num_modes = int(num_modes)
        self.num_candidates = self.num_commands * self.num_modes
        self.num_phase = int(num_phase)
        self.dt = float(dt)
        self.require_real_mamba = bool(require_real_mamba)
        self.allow_mamba_fallback = bool(allow_mamba_fallback)

        if self.horizon_steps != 12:
            raise ValueError("C3-A is defined for the audited 12-horizon MomAD-6s protocol")
        if self.num_phase != len(PHASE_NAMES):
            raise ValueError(f"C3-A expects {len(PHASE_NAMES)} phases")
        if self.allow_mamba_fallback:
            raise ValueError("C3-A forbids Horizon Mamba fallback")
        if self.require_real_mamba and RealMamba is None:
            raise ImportError("C3-A requires RealMamba; fallback is forbidden")

        if RealMamba is None:
            raise ImportError("RealMamba is unavailable and fallback is forbidden")
        self.horizon_mamba = RealMamba(
            d_model=self.token_dims,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.using_real_mamba = True
        self.fallback_activated = False

        self.plan_query_proj = nn.Linear(self.embed_dims, 64)
        self.memory_query_proj = nn.Linear(self.embed_dims, 64)
        self.geom_proj = nn.Linear(8, 48)
        self.horizon_embed = nn.Embedding(self.horizon_steps, 32)
        self.phase_embed = nn.Linear(self.num_phase, self.phase_embed_dims, bias=False)
        self.token_proj = nn.Sequential(
            nn.LayerNorm(64 + 64 + 48 + 32 + self.phase_embed_dims),
            nn.Linear(64 + 64 + 48 + 32 + self.phase_embed_dims, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, self.token_dims),
        )
        self.phase_head = nn.Sequential(
            nn.LayerNorm(self.embed_dims + 8 + 32),
            nn.Linear(self.embed_dims + 8 + 32, self.embed_dims),
            nn.GELU(),
            nn.Linear(self.embed_dims, self.num_phase),
        )
        self.gate_head = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, 1),
        )
        self.residual_head = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, 2),
        )
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        self.alpha_h = nn.Parameter(torch.ones(self.horizon_steps, 1))

    def _phase_forward(
        self,
        plan_query: torch.Tensor,
        geom: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query_context = plan_query.squeeze(1).mean(dim=1)
        geom_context = geom.mean(dim=(1, 2))
        h = torch.arange(self.horizon_steps, device=plan_query.device)
        h_embed = self.horizon_embed(h).unsqueeze(0).expand(plan_query.shape[0], -1, -1)
        context = query_context.unsqueeze(1).expand(-1, self.horizon_steps, -1)
        phase_input = torch.cat([context, geom_context, h_embed], dim=-1)
        phase_logits = self.phase_head(phase_input)
        phase_prob = torch.softmax(phase_logits, dim=-1)
        phase_embedding = self.phase_embed(phase_prob).view(
            plan_query.shape[0],
            1,
            1,
            self.horizon_steps,
            self.phase_embed_dims,
        )
        return phase_logits, phase_prob, phase_embedding

    def forward(
        self,
        coarse_cumulative_plan: torch.Tensor,
        plan_query: torch.Tensor,
        h1_memory_query: Optional[torch.Tensor] = None,
        h1_memory_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if coarse_cumulative_plan.dim() != 5:
            raise ValueError(
                "coarse_cumulative_plan must be [B,3,6,12,2], got "
                f"{tuple(coarse_cumulative_plan.shape)}"
            )
        b, c, m, t, xy = coarse_cumulative_plan.shape
        _assert_shape(
            coarse_cumulative_plan,
            (b, self.num_commands, self.num_modes, self.horizon_steps, 2),
            "coarse_cumulative_plan",
        )
        _assert_shape(plan_query, (b, 1, self.num_candidates, self.embed_dims), "plan_query")
        if xy != 2:
            raise ValueError("Trajectory final dimension must be xy=2")
        if h1_memory_query is None:
            h1_memory_query = plan_query.new_zeros(plan_query.shape)
        _assert_shape(
            h1_memory_query,
            (b, 1, self.num_candidates, self.embed_dims),
            "h1_memory_query",
        )
        if h1_memory_valid is not None:
            if h1_memory_valid.shape not in {(b,), (b, 1)}:
                raise ValueError(
                    "h1_memory_valid must be [B] or [B,1], got "
                    f"{tuple(h1_memory_valid.shape)}"
                )
            valid = h1_memory_valid.reshape(b, 1, 1, 1).to(
                device=plan_query.device,
                dtype=plan_query.dtype,
            )
            h1_memory_query = h1_memory_query * valid

        geom, vy, ay = trajectory_proxy_features(coarse_cumulative_plan, dt=self.dt)
        phase_logits, phase_prob, phase_embedding = self._phase_forward(plan_query, geom)

        query_tokens = self.plan_query_proj(plan_query.squeeze(1)).view(
            b, self.num_commands, self.num_modes, 1, 64
        )
        query_tokens = query_tokens.expand(-1, -1, -1, self.horizon_steps, -1)
        memory_tokens = self.memory_query_proj(h1_memory_query.squeeze(1)).view(
            b, self.num_commands, self.num_modes, 1, 64
        )
        memory_tokens = memory_tokens.expand(-1, -1, -1, self.horizon_steps, -1)
        geom_tokens = self.geom_proj(geom)
        h = torch.arange(self.horizon_steps, device=coarse_cumulative_plan.device)
        horizon_tokens = self.horizon_embed(h).view(1, 1, 1, self.horizon_steps, 32)
        horizon_tokens = horizon_tokens.expand(b, self.num_commands, self.num_modes, -1, -1)
        phase_tokens = phase_embedding.expand(
            b,
            self.num_commands,
            self.num_modes,
            self.horizon_steps,
            self.phase_embed_dims,
        )
        tokens = torch.cat(
            [query_tokens, memory_tokens, geom_tokens, horizon_tokens, phase_tokens],
            dim=-1,
        )
        tokens = self.token_proj(tokens)
        flat_tokens = tokens.reshape(b * self.num_candidates, self.horizon_steps, self.token_dims)
        mixed = self.horizon_mamba(flat_tokens)
        residual = self.residual_head(mixed)
        gate = torch.sigmoid(self.gate_head(mixed))
        alpha = self.alpha_h.view(1, self.horizon_steps, 1).to(
            device=residual.device,
            dtype=residual.dtype,
        )
        residual = residual * gate * alpha
        residual_cumulative = residual.reshape(
            b,
            self.num_commands,
            self.num_modes,
            self.horizon_steps,
            2,
        )
        refined_cumulative = coarse_cumulative_plan + residual_cumulative
        if not torch.isfinite(refined_cumulative).all():
            raise FloatingPointError("C3-A refined trajectory contains NaN/Inf")
        return {
            "refined_cumulative": refined_cumulative,
            "residual_cumulative": residual_cumulative,
            "phase_logits": phase_logits,
            "phase_prob": phase_prob,
            "gate": gate.reshape(
                b,
                self.num_commands,
                self.num_modes,
                self.horizon_steps,
                1,
            ),
            "vy": vy,
            "ay": ay,
        }


class C3AHorizonPlanningAdapter(nn.Module):
    """Adapter between MomAD [B,1,18,12,2] increments and C3-A horizon module."""

    def __init__(self, **kwargs):
        super().__init__()
        self.refiner = LongitudinalPhaseHorizonMamba(**kwargs)
        self.num_commands = self.refiner.num_commands
        self.num_modes = self.refiner.num_modes
        self.horizon_steps = self.refiner.horizon_steps

    @property
    def fallback_activated(self) -> bool:
        return bool(getattr(self.refiner, "fallback_activated", False))

    @property
    def using_real_mamba(self) -> bool:
        return bool(getattr(self.refiner, "using_real_mamba", False))

    def forward(
        self,
        plan_reg: torch.Tensor,
        plan_query: torch.Tensor,
        h1_memory_query: Optional[torch.Tensor] = None,
        h1_memory_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if plan_reg.dim() != 5:
            raise ValueError(f"plan_reg must be [B,1,18,12,2], got {tuple(plan_reg.shape)}")
        b = plan_reg.shape[0]
        _assert_shape(
            plan_reg,
            (b, 1, self.num_commands * self.num_modes, self.horizon_steps, 2),
            "plan_reg",
        )
        coarse_cumulative = plan_reg.squeeze(1).reshape(
            b,
            self.num_commands,
            self.num_modes,
            self.horizon_steps,
            2,
        ).cumsum(dim=-2)
        output = self.refiner(
            coarse_cumulative_plan=coarse_cumulative,
            plan_query=plan_query,
            h1_memory_query=h1_memory_query,
            h1_memory_valid=h1_memory_valid,
        )
        refined_increment = cumulative_to_increments(output["refined_cumulative"])
        output["refined_plan_reg"] = refined_increment.reshape(
            b,
            1,
            self.num_commands * self.num_modes,
            self.horizon_steps,
            2,
        )
        output["coarse_cumulative"] = coarse_cumulative
        return output
