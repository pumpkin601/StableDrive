"""Causal dense multi-objective scoring for C3 planning candidates.

The forward path is deliberately future-GT free.  Ground-truth trajectories
and future agent boxes are consumed only by :meth:`loss` to build dense labels
for all six candidates belonging to the commanded branch.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .retrospective_pareto_critic import avoidable_collision_proxy


CDRS_METRIC_NAMES = (
    "l2_4s",
    "l2_5s",
    "l2_6s",
    "avoidable_collision_4s",
    "avoidable_collision_5s",
    "avoidable_collision_6s",
    "consistency_4s",
    "consistency_5s",
    "consistency_6s",
    "terminal_progress_error",
)
HORIZON_STEPS = (8, 10, 12)
L2_INDICES = (0, 1, 2)
COLLISION_INDICES = (3, 4, 5)
CONSISTENCY_INDICES = (6, 7, 8)
TERMINAL_PROGRESS_INDEX = 9


def cumulative_to_increments(cumulative: torch.Tensor) -> torch.Tensor:
    if cumulative.shape[-1] != 2:
        raise ValueError("CDRS trajectories must end in xy=2")
    return torch.cat(
        [
            cumulative[..., :1, :],
            cumulative[..., 1:, :] - cumulative[..., :-1, :],
        ],
        dim=-2,
    )


def _batch_item(value, index: int, batch_size: int):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index]
    if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size:
        return value[index]
    return value


def _masked_horizon_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    value_h = value[..., :horizon]
    mask_h = mask[..., :horizon].to(device=value.device, dtype=value.dtype)
    denom = mask_h.sum(dim=-1).clamp_min(1.0)
    mean = (value_h * mask_h).sum(dim=-1) / denom
    return mean, mask_h.sum(dim=-1) > 0


class CausalDenseRiskScorer(nn.Module):
    """Score all C3 candidates using dense, causal multi-objective evidence.

    The score correction is an exact identity at initialization.  Dense metric
    prediction and a bounded residual learn candidate ranking, while explicit
    collision and Pareto losses prevent L2-only mode selection.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        token_dims: int = 192,
        horizon_steps: int = 12,
        num_commands: int = 3,
        num_modes: int = 6,
        num_heads: int = 6,
        dt: float = 0.5,
        dropout: float = 0.0,
        max_score_delta: float = 8.0,
        metric_score_scale: float = 4.0,
        residual_score_scale: float = 0.10,
        listwise_temperature: float = 0.20,
        pairwise_margin: float = 0.03,
        collision_rank_margin: float = 0.50,
        metric_loss_weight: float = 1.0,
        listwise_loss_weight: float = 1.0,
        pairwise_loss_weight: float = 0.25,
        collision_rank_loss_weight: float = 0.50,
        pareto_loss_weight: float = 0.25,
        identity_loss_weight: float = 0.01,
        metric_scales: Sequence[float] = (
            2.0,
            3.0,
            4.0,
            1.0,
            1.0,
            1.0,
            2.0,
            3.0,
            4.0,
            4.0,
        ),
        teacher_metric_weights: Sequence[float] = (
            0.10,
            0.15,
            0.25,
            0.06,
            0.09,
            0.15,
            0.04,
            0.06,
            0.10,
            0.00,
        ),
    ) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.token_dims = int(token_dims)
        self.horizon_steps = int(horizon_steps)
        self.num_commands = int(num_commands)
        self.num_modes = int(num_modes)
        self.num_candidates = self.num_commands * self.num_modes
        self.dt = float(dt)
        self.max_score_delta = float(max_score_delta)
        self.metric_score_scale = float(metric_score_scale)
        self.residual_score_scale = float(residual_score_scale)
        self.listwise_temperature = float(listwise_temperature)
        self.pairwise_margin = float(pairwise_margin)
        self.collision_rank_margin = float(collision_rank_margin)
        self.metric_loss_weight = float(metric_loss_weight)
        self.listwise_loss_weight = float(listwise_loss_weight)
        self.pairwise_loss_weight = float(pairwise_loss_weight)
        self.collision_rank_loss_weight = float(collision_rank_loss_weight)
        self.pareto_loss_weight = float(pareto_loss_weight)
        self.identity_loss_weight = float(identity_loss_weight)

        if self.horizon_steps != 12:
            raise ValueError("CDRS is locked to the audited 12-step/6-second task")
        if self.num_commands != 3 or self.num_modes != 6:
            raise ValueError("CDRS expects MomAD's 3 commands x 6 modes")
        if self.token_dims % int(num_heads) != 0:
            raise ValueError("token_dims must be divisible by num_heads")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        if self.max_score_delta <= 0.0:
            raise ValueError("max_score_delta must be positive")
        if self.listwise_temperature <= 0.0:
            raise ValueError("listwise_temperature must be positive")
        if self.pairwise_margin < 0.0 or self.collision_rank_margin <= 0.0:
            raise ValueError("ranking margins are invalid")
        loss_weights = (
            self.metric_loss_weight,
            self.listwise_loss_weight,
            self.pairwise_loss_weight,
            self.collision_rank_loss_weight,
            self.pareto_loss_weight,
            self.identity_loss_weight,
        )
        if min(loss_weights) < 0.0:
            raise ValueError("CDRS loss weights must be non-negative")
        if len(metric_scales) != len(CDRS_METRIC_NAMES):
            raise ValueError("metric_scales must contain ten values")
        if len(teacher_metric_weights) != len(CDRS_METRIC_NAMES):
            raise ValueError("teacher_metric_weights must contain ten values")
        metric_scales_tensor = torch.tensor(metric_scales, dtype=torch.float32)
        teacher_weights_tensor = torch.tensor(
            teacher_metric_weights, dtype=torch.float32
        )
        if (metric_scales_tensor <= 0).any():
            raise ValueError("metric scales must be positive")
        if (teacher_weights_tensor < 0).any() or teacher_weights_tensor.sum() <= 0:
            raise ValueError("teacher weights must be non-negative and non-zero")
        self.register_buffer("metric_scales", metric_scales_tensor, persistent=True)
        self.register_buffer(
            "teacher_metric_weights",
            teacher_weights_tensor / teacher_weights_tensor.sum(),
            persistent=True,
        )

        # Per-horizon geometry: xy, delta-xy, velocity-xy, acceleration-xy and
        # normalized horizon time.  No GT or future observation enters here.
        self.geometry_encoder = nn.Sequential(
            nn.Linear(9, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, self.token_dims),
        )
        self.horizon_embedding = nn.Embedding(self.horizon_steps, self.token_dims)
        self.horizon_attention = nn.MultiheadAttention(
            self.token_dims,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.horizon_norm = nn.LayerNorm(self.token_dims)
        self.plan_query_proj = nn.Linear(self.embed_dims, self.token_dims)
        self.agent_proj = nn.Linear(self.embed_dims, self.token_dims)
        self.map_proj = nn.Linear(self.embed_dims, self.token_dims)
        self.scene_attention = nn.MultiheadAttention(
            self.token_dims,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.scene_norm = nn.LayerNorm(self.token_dims)
        self.command_attention = nn.MultiheadAttention(
            self.token_dims,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.command_norm = nn.LayerNorm(self.token_dims)
        self.causal_context_encoder = nn.Sequential(
            nn.Linear(14, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, self.token_dims),
        )
        self.candidate_fusion = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.token_dims, self.token_dims),
        )
        self.candidate_norm = nn.LayerNorm(self.token_dims)

        self.metric_head = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, len(CDRS_METRIC_NAMES)),
        )
        self.pareto_head = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims // 2),
            nn.GELU(),
            nn.Linear(self.token_dims // 2, 1),
        )
        self.score_residual_head = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims // 2),
            nn.GELU(),
            nn.Linear(self.token_dims // 2, 1),
        )
        # Multiplying the interpretable metric cost by tanh(0)=0 preserves the
        # loaded C1 logits exactly before any CDRS update.
        self.metric_score_strength = nn.Parameter(torch.zeros(()))
        nn.init.zeros_(self.score_residual_head[-1].weight)
        nn.init.zeros_(self.score_residual_head[-1].bias)

    def _validate_forward_shapes(
        self,
        base_logits: torch.Tensor,
        plan_reg: torch.Tensor,
        plan_query: torch.Tensor,
        agent_features: torch.Tensor,
        map_features: torch.Tensor,
    ) -> int:
        if base_logits.dim() != 3 or tuple(base_logits.shape[1:]) != (
            1,
            self.num_candidates,
        ):
            raise ValueError(
                "base_logits must be [B,1,18], got %s"
                % (tuple(base_logits.shape),)
            )
        batch_size = base_logits.shape[0]
        expected_plan = (
            batch_size,
            1,
            self.num_candidates,
            self.horizon_steps,
            2,
        )
        if tuple(plan_reg.shape) != expected_plan:
            raise ValueError(
                "plan_reg must be %s, got %s"
                % (expected_plan, tuple(plan_reg.shape))
            )
        expected_query = (batch_size, 1, self.num_candidates, self.embed_dims)
        if tuple(plan_query.shape) != expected_query:
            raise ValueError(
                "plan_query must be %s, got %s"
                % (expected_query, tuple(plan_query.shape))
            )
        for name, value in (
            ("agent_features", agent_features),
            ("map_features", map_features),
        ):
            if value.dim() != 3 or value.shape[0] != batch_size:
                raise ValueError("%s must be [B,N,D]" % name)
            if value.shape[-1] != self.embed_dims:
                raise ValueError("%s feature dimension mismatch" % name)
        return batch_size

    def _trajectory_tokens(self, cumulative: torch.Tensor) -> torch.Tensor:
        batch_size, num_candidates, horizon, _ = cumulative.shape
        increments = cumulative_to_increments(cumulative)
        velocity = increments / self.dt
        acceleration = torch.zeros_like(velocity)
        acceleration[..., 1:, :] = (
            velocity[..., 1:, :] - velocity[..., :-1, :]
        ) / self.dt
        time = torch.linspace(
            1.0 / self.horizon_steps,
            1.0,
            self.horizon_steps,
            device=cumulative.device,
            dtype=cumulative.dtype,
        ).view(1, 1, self.horizon_steps, 1)
        time = time.expand(batch_size, num_candidates, -1, -1)
        geometry = torch.cat(
            [cumulative, increments, velocity, acceleration, time], dim=-1
        )
        tokens = self.geometry_encoder(geometry)
        horizon_index = torch.arange(horizon, device=cumulative.device)
        tokens = tokens + self.horizon_embedding(horizon_index).view(
            1, 1, horizon, self.token_dims
        )
        flat = tokens.reshape(batch_size * num_candidates, horizon, self.token_dims)
        update, _ = self.horizon_attention(flat, flat, flat, need_weights=False)
        flat = self.horizon_norm(flat + update)
        tokens = flat.reshape(batch_size, num_candidates, horizon, self.token_dims)
        return 0.5 * (tokens.mean(dim=2) + tokens[:, :, -1])

    def _causal_context(
        self,
        cumulative: torch.Tensor,
        last_final_planning: Optional[torch.Tensor],
        history_valid: Optional[torch.Tensor],
        ego_status: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = cumulative.shape[0]
        if last_final_planning is None:
            last_final_planning = cumulative.new_zeros(
                batch_size, self.horizon_steps, 2
            )
        else:
            last_final_planning = last_final_planning.to(
                device=cumulative.device, dtype=cumulative.dtype
            )
        if history_valid is None:
            history_valid = torch.zeros(
                batch_size, device=cumulative.device, dtype=torch.bool
            )
        else:
            history_valid = history_valid.to(device=cumulative.device).bool()
        distance = torch.linalg.norm(
            cumulative - last_final_planning[:, None], dim=-1
        )
        consistency = []
        for metric_index, horizon in zip(CONSISTENCY_INDICES, HORIZON_STEPS):
            value = distance[..., :horizon].mean(dim=-1)
            scale = self.metric_scales[metric_index].to(
                device=cumulative.device, dtype=cumulative.dtype
            )
            consistency.append(torch.clamp(value / scale, 0.0, 5.0))
        analytic_consistency = torch.stack(consistency, dim=-1)
        analytic_consistency = analytic_consistency * history_valid.to(
            cumulative.dtype
        ).view(batch_size, 1, 1)

        if ego_status is None:
            ego = cumulative.new_zeros(batch_size, 10)
        else:
            ego = ego_status.to(device=cumulative.device, dtype=cumulative.dtype)
            ego = ego.reshape(batch_size, -1)
            if ego.shape[-1] < 10:
                ego = F.pad(ego, (0, 10 - ego.shape[-1]))
            ego = ego[:, :10]
        ego = torch.clamp(ego / 20.0, -5.0, 5.0)
        history = history_valid.to(cumulative.dtype).view(batch_size, 1, 1)
        context_input = torch.cat(
            [
                analytic_consistency,
                history.expand(-1, self.num_candidates, -1),
                ego[:, None].expand(-1, self.num_candidates, -1),
            ],
            dim=-1,
        )
        return self.causal_context_encoder(context_input), analytic_consistency

    def forward(
        self,
        base_logits: torch.Tensor,
        plan_reg: torch.Tensor,
        plan_query: torch.Tensor,
        agent_features: torch.Tensor,
        map_features: torch.Tensor,
        last_final_planning: Optional[torch.Tensor] = None,
        history_valid: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = self._validate_forward_shapes(
            base_logits, plan_reg, plan_query, agent_features, map_features
        )
        cumulative = plan_reg.squeeze(1).cumsum(dim=-2)
        candidate = self._trajectory_tokens(cumulative)
        candidate = candidate + self.plan_query_proj(plan_query.squeeze(1))

        scene_tokens = torch.cat(
            [self.agent_proj(agent_features), self.map_proj(map_features)], dim=1
        )
        scene_update, _ = self.scene_attention(
            candidate, scene_tokens, scene_tokens, need_weights=False
        )
        candidate = self.scene_norm(candidate + scene_update)
        causal_context, analytic_consistency = self._causal_context(
            cumulative, last_final_planning, history_valid, ego_status
        )
        candidate = candidate + causal_context
        command_candidate = candidate.reshape(
            batch_size * self.num_commands, self.num_modes, self.token_dims
        )
        relative_update, _ = self.command_attention(
            command_candidate,
            command_candidate,
            command_candidate,
            need_weights=False,
        )
        command_candidate = self.command_norm(command_candidate + relative_update)
        candidate = command_candidate.reshape(
            batch_size, self.num_candidates, self.token_dims
        )
        candidate = self.candidate_norm(candidate + self.candidate_fusion(candidate))

        metric_raw = self.metric_head(candidate)
        predicted_metrics = F.softplus(metric_raw)
        collision_index = torch.as_tensor(
            COLLISION_INDICES, device=metric_raw.device, dtype=torch.long
        )
        predicted_metrics = predicted_metrics.scatter(
            -1,
            collision_index.view(1, 1, -1).expand(batch_size, self.num_candidates, -1),
            torch.sigmoid(metric_raw[..., list(COLLISION_INDICES)]),
        )
        # Consistency is exactly observable from causal H1 geometry.  It is
        # still densely supervised, while the score uses the analytic value to
        # avoid learning an approximation to an available signal.
        score_metrics = predicted_metrics.clone()
        score_metrics[..., list(CONSISTENCY_INDICES)] = analytic_consistency
        weights = self.teacher_metric_weights.to(
            device=cumulative.device, dtype=cumulative.dtype
        )
        metric_cost = (score_metrics * weights.view(1, 1, -1)).sum(dim=-1)
        metric_cost_by_command = metric_cost.reshape(
            batch_size, self.num_commands, self.num_modes
        )
        centered_cost = metric_cost_by_command - metric_cost_by_command.mean(
            dim=-1, keepdim=True
        )
        metric_delta = (
            -self.metric_score_scale
            * torch.tanh(self.metric_score_strength)
            * centered_cost
        )
        residual_delta = self.residual_score_scale * torch.tanh(
            self.score_residual_head(candidate).squeeze(-1)
        ).reshape(batch_size, self.num_commands, self.num_modes)
        score_delta = torch.clamp(
            metric_delta + residual_delta,
            min=-self.max_score_delta,
            max=self.max_score_delta,
        ).reshape(batch_size, 1, self.num_candidates)
        calibrated_logits = base_logits + score_delta
        pareto_logits = self.pareto_head(candidate).squeeze(-1)

        output = {
            "base_logits": base_logits,
            "calibrated_logits": calibrated_logits,
            "score_delta": score_delta,
            "metric_raw": metric_raw,
            "predicted_metrics": predicted_metrics,
            "score_metrics": score_metrics,
            "metric_cost": metric_cost,
            "pareto_logits": pareto_logits,
            "candidate_features": candidate,
            "candidate_cumulative": cumulative,
            "history_reference": (
                cumulative.new_zeros(batch_size, self.horizon_steps, 2)
                if last_final_planning is None
                else last_final_planning.detach().to(
                    device=cumulative.device, dtype=cumulative.dtype
                )
            ),
            "history_valid": (
                torch.zeros(batch_size, device=cumulative.device, dtype=torch.bool)
                if history_valid is None
                else history_valid.detach().to(device=cumulative.device).bool()
            ),
            "refined_plan_reg": plan_reg,
        }
        for name, value in output.items():
            if torch.is_tensor(value) and value.is_floating_point():
                if not torch.isfinite(value).all():
                    raise FloatingPointError("CDRS output %s contains NaN/Inf" % name)
        return output

    @torch.no_grad()
    def build_targets(
        self,
        output: Dict[str, torch.Tensor],
        data: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        cumulative = output["candidate_cumulative"].detach()
        batch_size = cumulative.shape[0]
        command = data["gt_ego_fut_cmd"].argmax(dim=-1).long()
        batch_index = torch.arange(batch_size, device=cumulative.device)
        candidate = cumulative.reshape(
            batch_size,
            self.num_commands,
            self.num_modes,
            self.horizon_steps,
            2,
        )[batch_index, command]
        gt_increment = data["gt_ego_fut_trajs"].to(
            device=cumulative.device, dtype=cumulative.dtype
        )
        gt_mask = data["gt_ego_fut_masks"].to(device=cumulative.device).bool()
        gt_cumulative = gt_increment.cumsum(dim=-2)
        distance = torch.linalg.norm(candidate - gt_cumulative[:, None], dim=-1)
        target = cumulative.new_zeros(
            batch_size, self.num_modes, len(CDRS_METRIC_NAMES)
        )
        valid = torch.zeros_like(target, dtype=torch.bool)

        expanded_mask = gt_mask[:, None].expand(-1, self.num_modes, -1)
        for metric_index, horizon in zip(L2_INDICES, HORIZON_STEPS):
            value, horizon_valid = _masked_horizon_mean(
                distance, expanded_mask, horizon
            )
            target[..., metric_index] = torch.clamp(
                value
                / self.metric_scales[metric_index].to(
                    device=cumulative.device, dtype=cumulative.dtype
                ),
                0.0,
                5.0,
            )
            valid[..., metric_index] = horizon_valid

        collision_available = []
        for sample_index in range(batch_size):
            sample_available = True
            for metric_index, horizon in zip(COLLISION_INDICES, HORIZON_STEPS):
                collision, available = avoidable_collision_proxy(
                    candidate[sample_index, :, :horizon],
                    gt_cumulative[sample_index, :horizon],
                    gt_mask[sample_index, :horizon],
                    _batch_item(data.get("gt_bboxes_3d"), sample_index, batch_size),
                    _batch_item(
                        data.get("gt_agent_fut_trajs"), sample_index, batch_size
                    ),
                    _batch_item(
                        data.get("gt_agent_fut_masks"), sample_index, batch_size
                    ),
                )
                target[sample_index, :, metric_index] = collision
                valid[sample_index, :, metric_index] = bool(available)
                sample_available = sample_available and bool(available)
            collision_available.append(sample_available)

        history_reference = output["history_reference"].detach()
        history_valid = output["history_valid"].detach().bool()
        consistency_distance = torch.linalg.norm(
            candidate - history_reference[:, None], dim=-1
        )
        for metric_index, horizon in zip(CONSISTENCY_INDICES, HORIZON_STEPS):
            value, horizon_valid = _masked_horizon_mean(
                consistency_distance, expanded_mask, horizon
            )
            target[..., metric_index] = torch.clamp(
                value
                / self.metric_scales[metric_index].to(
                    device=cumulative.device, dtype=cumulative.dtype
                ),
                0.0,
                5.0,
            )
            valid[..., metric_index] = horizon_valid & history_valid[:, None]

        gt_endpoint = gt_cumulative[:, -1]
        gt_progress = torch.linalg.norm(gt_endpoint, dim=-1).clamp_min(1e-4)
        gt_direction = gt_endpoint / gt_progress[:, None]
        candidate_progress = (
            candidate[:, :, -1] * gt_direction[:, None]
        ).sum(dim=-1)
        progress_error = (candidate_progress - gt_progress[:, None]).abs()
        target[..., TERMINAL_PROGRESS_INDEX] = torch.clamp(
            progress_error
            / self.metric_scales[TERMINAL_PROGRESS_INDEX].to(
                device=cumulative.device, dtype=cumulative.dtype
            ),
            0.0,
            5.0,
        )
        valid[..., TERMINAL_PROGRESS_INDEX] = gt_mask[:, -1, None]

        effective_weight = self.teacher_metric_weights.to(
            device=cumulative.device, dtype=cumulative.dtype
        ).view(1, 1, -1) * valid.to(cumulative.dtype)
        effective_weight = effective_weight / effective_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        teacher_cost = (target * effective_weight).sum(dim=-1)
        rank_valid = valid[..., list(L2_INDICES)].all(dim=-1).all(dim=-1)

        # A candidate is Pareto-positive when no other mode is no-worse in all
        # nine evaluation cells and strictly better in at least one.
        eval_target = target[..., :9]
        eval_valid = valid[..., :9].all(dim=-1)
        left = eval_target[:, :, None, :]
        right = eval_target[:, None, :, :]
        other_dominates = (right <= left + 1e-8).all(dim=-1) & (
            right < left - 1e-8
        ).any(dim=-1)
        eye = torch.eye(
            self.num_modes, device=cumulative.device, dtype=torch.bool
        ).unsqueeze(0)
        other_dominates = other_dominates & (~eye)
        pareto_target = (~other_dominates.any(dim=-1)) & eval_valid
        return {
            "metric_target": target,
            "metric_valid": valid,
            "teacher_cost": teacher_cost,
            "rank_valid": rank_valid,
            "pareto_target": pareto_target,
            "pareto_valid": eval_valid & rank_valid[:, None],
            "command": command,
            "collision_available": torch.tensor(
                collision_available, device=cumulative.device, dtype=torch.bool
            ),
        }

    def loss(
        self,
        output: Dict[str, torch.Tensor],
        data: Dict[str, object],
        targets: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        if targets is None:
            targets = self.build_targets(output, data)
        base = output["base_logits"]
        batch_size = base.shape[0]
        batch_index = torch.arange(batch_size, device=base.device)
        command = targets["command"]
        predicted = output["predicted_metrics"].reshape(
            batch_size,
            self.num_commands,
            self.num_modes,
            len(CDRS_METRIC_NAMES),
        )[batch_index, command]
        metric_raw = output["metric_raw"].reshape(
            batch_size,
            self.num_commands,
            self.num_modes,
            len(CDRS_METRIC_NAMES),
        )[batch_index, command]
        target = targets["metric_target"]
        valid = targets["metric_valid"]

        regression_indices = list(L2_INDICES + CONSISTENCY_INDICES) + [
            TERMINAL_PROGRESS_INDEX
        ]
        regression_valid = valid[..., regression_indices].to(base.dtype)
        regression_error = F.smooth_l1_loss(
            predicted[..., regression_indices],
            target[..., regression_indices],
            reduction="none",
            beta=0.10,
        )
        regression_loss = (
            regression_error * regression_valid
        ).sum() / regression_valid.sum().clamp_min(1.0)

        collision_valid = valid[..., list(COLLISION_INDICES)].to(base.dtype)
        collision_bce = F.binary_cross_entropy_with_logits(
            metric_raw[..., list(COLLISION_INDICES)],
            target[..., list(COLLISION_INDICES)],
            reduction="none",
        )
        collision_positive_weight = 1.0 + 7.0 * target[
            ..., list(COLLISION_INDICES)
        ]
        collision_loss = (
            collision_bce * collision_positive_weight * collision_valid
        ).sum() / (
            collision_positive_weight * collision_valid
        ).sum().clamp_min(1.0)
        metric_loss = regression_loss + collision_loss

        score = output.get("ranking_logits", output["calibrated_logits"]).reshape(
            batch_size, self.num_commands, self.num_modes
        )[batch_index, command]
        teacher_cost = targets["teacher_cost"]
        teacher_probability = torch.softmax(
            -teacher_cost / self.listwise_temperature, dim=-1
        )
        listwise_per_sample = -(
            teacher_probability * F.log_softmax(score, dim=-1)
        ).sum(dim=-1)
        rank_valid_float = targets["rank_valid"].to(base.dtype)
        listwise_loss = (
            listwise_per_sample * rank_valid_float
        ).sum() / rank_valid_float.sum().clamp_min(1.0)

        cost_difference = teacher_cost[:, None, :] - teacher_cost[:, :, None]
        preference = torch.sign(cost_difference)
        score_difference = score[:, :, None] - score[:, None, :]
        pair_mask = torch.triu(
            torch.ones(
                self.num_modes,
                self.num_modes,
                device=base.device,
                dtype=torch.bool,
            ),
            diagonal=1,
        ).unsqueeze(0)
        pair_mask = (
            pair_mask
            & (cost_difference.abs() > self.pairwise_margin)
            & targets["rank_valid"][:, None, None]
        )
        pair_value = F.softplus(-preference * score_difference)
        pair_mask_float = pair_mask.to(base.dtype)
        pairwise_loss = (
            pair_value * pair_mask_float
        ).sum() / pair_mask_float.sum().clamp_min(1.0)

        collision_target = target[..., list(COLLISION_INDICES)].max(dim=-1).values
        collision_target_valid = valid[..., list(COLLISION_INDICES)].all(dim=-1)
        safe = (collision_target <= 1e-8) & collision_target_valid
        unsafe = (collision_target > 1e-8) & collision_target_valid
        collision_pair_valid = (
            safe[:, :, None]
            & unsafe[:, None, :]
            & targets["rank_valid"][:, None, None]
        )
        safe_minus_unsafe = score[:, :, None] - score[:, None, :]
        collision_rank_value = F.softplus(
            self.collision_rank_margin - safe_minus_unsafe
        )
        collision_pair_float = collision_pair_valid.to(base.dtype)
        collision_rank_loss = (
            collision_rank_value * collision_pair_float
        ).sum() / collision_pair_float.sum().clamp_min(1.0)

        pareto_logits = output["pareto_logits"].reshape(
            batch_size, self.num_commands, self.num_modes
        )[batch_index, command]
        pareto_valid = targets["pareto_valid"].to(base.dtype)
        pareto_target = targets["pareto_target"].to(base.dtype)
        pareto_bce = F.binary_cross_entropy_with_logits(
            pareto_logits, pareto_target, reduction="none"
        )
        pareto_positive_weight = 1.0 + 3.0 * pareto_target
        pareto_loss = (
            pareto_bce * pareto_positive_weight * pareto_valid
        ).sum() / (
            pareto_positive_weight * pareto_valid
        ).sum().clamp_min(1.0)

        score_delta = output.get("ranking_score_delta", output["score_delta"])
        identity_loss = score_delta.square().mean()
        losses = {
            "cdrs_loss_metric": metric_loss * self.metric_loss_weight,
            "cdrs_loss_listwise": listwise_loss * self.listwise_loss_weight,
            "cdrs_loss_pairwise": pairwise_loss * self.pairwise_loss_weight,
            "cdrs_loss_collision_rank": (
                collision_rank_loss * self.collision_rank_loss_weight
            ),
            "cdrs_loss_pareto": pareto_loss * self.pareto_loss_weight,
            "cdrs_loss_identity": identity_loss * self.identity_loss_weight,
            "cdrs_stat_collision_target_coverage": targets[
                "collision_available"
            ].to(base.dtype).mean().detach(),
            "cdrs_stat_pareto_positive_rate": (
                (pareto_target * pareto_valid).sum()
                / pareto_valid.sum().clamp_min(1.0)
            ).detach(),
            "cdrs_stat_score_delta_abs_mean": score_delta.abs().mean().detach(),
        }
        for name, value in losses.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError("CDRS loss %s contains NaN/Inf" % name)
        return losses


class CDRSPlanningAdapter(CausalDenseRiskScorer):
    """Configuration-facing alias used by the MomAD planning head."""
