from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn

try:
    from mamba_ssm.modules.mamba_simple import Mamba as RealMamba
except ImportError:
    RealMamba = None


class SelectiveSSMFallback(nn.Module):
    """Fallback selective SSM-style block for interface and shape validation.

    This is intentionally not official Mamba. It uses depthwise temporal
    convolution, gating, and an MLP to mimic a lightweight sequence mixer when
    `mamba_ssm` is unavailable.
    """

    def __init__(self, embed_dims: int, kernel_size: int = 3, mlp_ratio: int = 2):
        super().__init__()
        padding = kernel_size // 2
        hidden_dims = embed_dims * mlp_ratio

        self.pre_norm = nn.LayerNorm(embed_dims)
        self.depthwise_conv = nn.Conv1d(
            embed_dims,
            embed_dims,
            kernel_size=kernel_size,
            padding=padding,
            groups=embed_dims,
        )
        self.gate = nn.Linear(embed_dims, embed_dims)
        self.channel_proj = nn.Linear(embed_dims, embed_dims)
        self.mlp_norm = nn.LayerNorm(embed_dims)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dims, hidden_dims),
            nn.GELU(),
            nn.Linear(hidden_dims, embed_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.pre_norm(x)
        conv_feat = self.depthwise_conv(x.transpose(1, 2)).transpose(1, 2)
        gated_feat = conv_feat * torch.sigmoid(self.gate(x))
        mixed = self.channel_proj(gated_feat)
        mixed = mixed + self.mlp(self.mlp_norm(mixed))
        return residual + mixed


class SelectiveMomentumMemory(nn.Module):
    def __init__(
        self,
        embed_dims: int = 256,
        num_modes: int = 18,
        history_len: int = 1,
        use_real_mamba: bool = True,
        fallback_if_unavailable: bool = True,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.embed_dims = embed_dims
        self.num_modes = num_modes
        self.history_len = history_len
        self.use_real_mamba = use_real_mamba
        self.fallback_if_unavailable = fallback_if_unavailable

        if use_real_mamba and RealMamba is not None:
            self.sequence_model = RealMamba(
                d_model=embed_dims,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self.using_real_mamba = True
        elif fallback_if_unavailable:
            self.sequence_model = SelectiveSSMFallback(embed_dims=embed_dims)
            self.using_real_mamba = False
        else:
            raise ImportError(
                "mamba_ssm is unavailable and fallback_if_unavailable=False."
            )

        self.output_norm = nn.LayerNorm(embed_dims)
        # Use a tiny non-zero residual gate so SMM parameters stay connected to
        # the loss graph from the first valid training step while remaining
        # close to identity behavior.
        self.alpha = nn.Parameter(torch.full((1,), 1e-3))
        self.debug_enabled = os.environ.get("MAMBA_MPI_DEBUG", "0").lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        self.debug_max_steps = int(os.environ.get("MAMBA_MPI_DEBUG_MAX_STEPS", "2"))
        self._debug_forward_idx = 0
        self._debug_active_step = 0
        self._debug_grad_stats = {}
        self._debug_last_report_step = 0
        self._debug_param_names = [name for name, _ in self.named_parameters()]
        self.last_group_records = []
        if self.debug_enabled:
            for name, param in self.named_parameters():
                param.register_hook(self._make_param_grad_hook(name))

    def _make_param_grad_hook(self, name: str):
        def hook(grad: torch.Tensor) -> torch.Tensor:
            if self.debug_enabled and self._debug_active_step <= self.debug_max_steps:
                grad_norm = float(grad.detach().norm().cpu())
                self._debug_grad_stats[name] = dict(is_none=False, norm=grad_norm)
                print(
                    f"[MAMBA_MPI_DEBUG][SMM][backward][step={self._debug_active_step}] "
                    f"param={name} grad_is_none=False grad_norm={grad_norm:.6e}"
                )
            return grad

        return hook

    def _debug_print(self, message: str, step: Optional[int] = None) -> None:
        if not self.debug_enabled:
            return
        if step is None:
            step = self._debug_active_step
        if step > self.debug_max_steps:
            return
        print(f"[MAMBA_MPI_DEBUG][SMM][step={step}] {message}")

    def _debug_report_last_step_grads(self) -> None:
        if not self.debug_enabled:
            return
        step = self._debug_forward_idx - 1
        if step <= 0 or step == self._debug_last_report_step or step > self.debug_max_steps:
            return
        for name in self._debug_param_names:
            stat = self._debug_grad_stats.get(name, dict(is_none=True, norm=None))
            print(
                f"[MAMBA_MPI_DEBUG][SMM][grad-summary][step={step}] "
                f"param={name} requires_grad=True grad_is_none={stat['is_none']} "
                f"grad_norm={stat['norm']}"
            )
        self._debug_last_report_step = step

    def _ensure_4d_query(self, tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        if tensor.dim() == 3:
            return tensor.unsqueeze(1)
        if tensor.dim() != 4:
            raise ValueError(f"Expected 4D query tensor, got shape {tuple(tensor.shape)}")
        return tensor

    def _ensure_3d_score(
        self,
        history_score: Optional[torch.Tensor],
        history_query: torch.Tensor,
    ) -> torch.Tensor:
        if history_score is None:
            return history_query.new_zeros(history_query.shape[:3])
        if history_score.dim() == 2:
            history_score = history_score.unsqueeze(1)
        if history_score.dim() != 3:
            raise ValueError(
                f"Expected 3D history score tensor, got shape {tuple(history_score.shape)}"
            )
        return history_score

    def _ensure_2d_history_mask(
        self,
        history_valid_mask: Optional[torch.Tensor],
        history_query: torch.Tensor,
    ) -> torch.Tensor:
        if history_valid_mask is None:
            return history_query.abs().sum(dim=(2, 3)) > 0
        if history_valid_mask.dim() == 3 and history_valid_mask.shape[-1] == 1:
            history_valid_mask = history_valid_mask.squeeze(-1)
        if history_valid_mask.dim() != 2:
            raise ValueError(
                "Expected 2D history valid mask tensor, got shape "
                f"{tuple(history_valid_mask.shape)}"
        )
        return history_valid_mask.to(device=history_query.device, dtype=torch.bool)

    def _ensure_2d_history_beta(
        self,
        history_beta: Optional[torch.Tensor],
        history_query: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if history_beta is None:
            return None
        if history_beta.dim() == 3 and history_beta.shape[-1] == 1:
            history_beta = history_beta.squeeze(-1)
        if history_beta.dim() != 2:
            raise ValueError(
                "Expected 2D history beta tensor, got shape "
                f"{tuple(history_beta.shape)}"
            )
        if history_beta.shape != history_query.shape[:2]:
            raise ValueError(
                "history_beta shape is incompatible with history_query: "
                f"{tuple(history_beta.shape)} vs {tuple(history_query.shape)}"
            )
        return history_beta.to(device=history_query.device, dtype=history_query.dtype)

    def _ensure_2d_history_weight(
        self,
        history_weight: Optional[torch.Tensor],
        history_query: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        if history_weight is None:
            return None
        if history_weight.dim() == 3 and history_weight.shape[-1] == 1:
            history_weight = history_weight.squeeze(-1)
        if history_weight.dim() != 2:
            raise ValueError(
                "Expected 2D history weight tensor, got shape "
                f"{tuple(history_weight.shape)}"
            )
        if history_weight.shape != history_query.shape[:2]:
            raise ValueError(
                "history_weight shape is incompatible with history_query: "
                f"{tuple(history_weight.shape)} vs {tuple(history_query.shape)}"
            )
        return history_weight.to(device=history_query.device, dtype=history_query.dtype)

    def _zero_parameter_dependency(self, reference: torch.Tensor) -> torch.Tensor:
        zero = reference.new_zeros(())
        for param in self.parameters():
            zero = zero + param.to(dtype=reference.dtype).sum() * 0.0
        return zero

    def forward(
        self,
        current_query: torch.Tensor,
        history_query: Optional[torch.Tensor] = None,
        history_score: Optional[torch.Tensor] = None,
        history_valid_mask: Optional[torch.Tensor] = None,
        history_beta: Optional[torch.Tensor] = None,
        history_weight: Optional[torch.Tensor] = None,
        allocation_audit: Optional[dict] = None,
    ) -> torch.Tensor:
        self._debug_report_last_step_grads()
        self._debug_forward_idx += 1
        self._debug_active_step = self._debug_forward_idx
        self._debug_grad_stats = {
            name: dict(is_none=True, norm=None) for name in self._debug_param_names
        }
        current_query = self._ensure_4d_query(current_query)
        batch_size, _, num_modes, channels = current_query.shape
        history_query = self._ensure_4d_query(history_query)
        self._debug_print(
            "forward called "
            f"current_query_shape={tuple(current_query.shape)} "
            f"current_query_requires_grad={current_query.requires_grad} "
            f"history_query_shape={None if history_query is None else tuple(history_query.shape)} "
            f"alpha={float(self.alpha.detach().cpu()):.6e} "
            f"alpha_requires_grad={self.alpha.requires_grad}"
        )
        if history_query is None or history_query.numel() == 0:
            self._debug_print("history_query missing or empty, injecting zero history")
            history_query = current_query.new_zeros(batch_size, 1, num_modes, channels)
            history_score = current_query.new_zeros(batch_size, 1, num_modes)
            history_valid_mask = torch.zeros(
                batch_size, 1, device=current_query.device, dtype=torch.bool
            )

        history_score = self._ensure_3d_score(history_score, history_query)
        history_valid_mask = self._ensure_2d_history_mask(
            history_valid_mask, history_query
        )
        history_beta = self._ensure_2d_history_beta(history_beta, history_query)
        history_weight = self._ensure_2d_history_weight(history_weight, history_query)
        if history_beta is not None and history_weight is not None:
            raise ValueError("history_beta and history_weight are mutually exclusive")
        effective_history_weight = history_weight if history_weight is not None else history_beta
        history_weight_source = (
            "history_weight"
            if history_weight is not None
            else "history_beta"
            if history_beta is not None
            else None
        )
        if channels != self.embed_dims:
            raise ValueError(
                f"Expected embed_dims={self.embed_dims}, got current_query.shape[-1]={channels}"
            )
        if num_modes != self.num_modes:
            raise ValueError(
                f"Expected num_modes={self.num_modes}, got current_query.shape[-2]={num_modes}"
            )

        if history_query.shape[0] != batch_size or history_query.shape[2:] != (num_modes, channels):
            raise ValueError(
                "history_query shape is incompatible with current_query: "
                f"{tuple(history_query.shape)} vs {tuple(current_query.shape)}"
            )
        if history_score.shape[:2] != history_query.shape[:2] or history_score.shape[2] != num_modes:
            raise ValueError(
                "history_score shape is incompatible with history_query: "
                f"{tuple(history_score.shape)} vs {tuple(history_query.shape)}"
            )
        if history_valid_mask.shape != history_query.shape[:2]:
            raise ValueError(
                "history_valid_mask shape is incompatible with history_query: "
                f"{tuple(history_valid_mask.shape)} vs {tuple(history_query.shape)}"
            )

        if self.history_len > 0 and history_query.shape[1] > self.history_len:
            history_query = history_query[:, -self.history_len :]
            history_score = history_score[:, -self.history_len :]
            history_valid_mask = history_valid_mask[:, -self.history_len :]
            if history_beta is not None:
                history_beta = history_beta[:, -self.history_len :]
            if history_weight is not None:
                history_weight = history_weight[:, -self.history_len :]
            effective_history_weight = (
                history_weight if history_weight is not None else history_beta
            )

        valid_counts = history_valid_mask.long().sum(dim=1)
        self.last_group_records = []
        self._debug_print(
            f"history_valid_count_min={int(valid_counts.min().item())} "
            f"history_valid_count_max={int(valid_counts.max().item())}"
        )
        has_valid_history = bool(torch.any(valid_counts > 0).item())
        if has_valid_history:
            output = current_query
        else:
            output = current_query + self._zero_parameter_dependency(current_query)
        for valid_len_tensor in torch.unique(valid_counts):
            valid_len = int(valid_len_tensor.item())
            if valid_len <= 0:
                continue
            batch_indices = (valid_counts == valid_len).nonzero(as_tuple=True)[0]
            current_group = current_query.index_select(0, batch_indices)
            history_group = history_query.index_select(0, batch_indices)[:, -valid_len:]
            score_group = history_score.index_select(0, batch_indices)[:, -valid_len:]
            score_sigmoid = torch.sigmoid(score_group)
            score_weighted_history = history_group * score_sigmoid.unsqueeze(-1)
            weighted_history = score_weighted_history
            beta_shape = None
            beta_stats = None
            reconstruction_error = None
            history_weight_shape = None
            history_weight_stats = None
            final_modulation_records = []
            low_score_threshold = 0.25
            low_score_stale_w_gt_1_count = 0
            stale_w_gt_1_count = 0
            if effective_history_weight is not None:
                weight_group = effective_history_weight.index_select(0, batch_indices)[:, -valid_len:]
                beta_shape = list(weight_group.shape) if history_weight_source == "history_beta" else None
                history_weight_shape = list(weight_group.shape)
                weight_expanded = weight_group[:, :, None, None]
                weighted_history = weighted_history * weight_expanded
                reconstructed = history_group * score_sigmoid.unsqueeze(-1) * weight_expanded
                reconstruction_error = float(
                    (weighted_history.detach().float() - reconstructed.detach().float())
                    .abs()
                    .max()
                    .cpu()
                    .item()
                )
                weight_detached = weight_group.detach().float()
                stats_payload = {
                    "count": int(weight_detached.numel()),
                    "mean": float(weight_detached.mean().cpu().item()) if weight_detached.numel() else None,
                    "std": float(weight_detached.std(unbiased=False).cpu().item()) if weight_detached.numel() else None,
                    "min": float(weight_detached.min().cpu().item()) if weight_detached.numel() else None,
                    "max": float(weight_detached.max().cpu().item()) if weight_detached.numel() else None,
                }
                if history_weight_source == "history_beta":
                    beta_stats = dict(stats_payload)
                else:
                    history_weight_stats = dict(stats_payload)
                final_modulation = score_sigmoid.detach().float() * weight_detached[:, :, None]
                score_group_detached = score_group.detach().float()
                score_sigmoid_slot = score_sigmoid.detach().float()
                for local_batch_idx in range(weight_detached.shape[0]):
                    for local_slot_idx in range(valid_len):
                        age_i = int(valid_len - local_slot_idx)
                        slot_scores = score_group_detached[local_batch_idx, local_slot_idx]
                        slot_sigmoid = score_sigmoid_slot[local_batch_idx, local_slot_idx]
                        slot_final = final_modulation[local_batch_idx, local_slot_idx]
                        final_modulation_records.append(
                            {
                                "sample_in_group": int(local_batch_idx),
                                "slot_index_within_valid_history": int(local_slot_idx),
                                "age_i": age_i,
                                "valid_mask_i": True,
                                "history_score_logit_mean": float(slot_scores.mean().cpu().item()),
                                "history_score_logit_min": float(slot_scores.min().cpu().item()),
                                "history_score_logit_max": float(slot_scores.max().cpu().item()),
                                "sigmoid_history_score_mean": float(slot_sigmoid.mean().cpu().item()),
                                "sigmoid_history_score_min": float(slot_sigmoid.min().cpu().item()),
                                "sigmoid_history_score_max": float(slot_sigmoid.max().cpu().item()),
                                "competitive_weight_w_i": float(
                                    weight_detached[local_batch_idx, local_slot_idx].cpu().item()
                                ),
                                "final_modulation_mean": float(slot_final.mean().cpu().item()),
                                "final_modulation_min": float(slot_final.min().cpu().item()),
                                "final_modulation_max": float(slot_final.max().cpu().item()),
                            }
                        )
                        if age_i >= 3 and float(
                            weight_detached[local_batch_idx, local_slot_idx].cpu().item()
                        ) > 1.0:
                            stale_w_gt_1_count += 1
                            if float(slot_sigmoid.mean().cpu().item()) < low_score_threshold:
                                low_score_stale_w_gt_1_count += 1
            else:
                reconstructed = history_group * score_sigmoid.unsqueeze(-1)
                reconstruction_error = float(
                    (weighted_history.detach().float() - reconstructed.detach().float())
                    .abs()
                    .max()
                    .cpu()
                    .item()
                )
            score_sigmoid_detached = score_sigmoid.detach().float()
            weighted_history_detached = weighted_history.detach().float()
            score_weighted_history_detached = score_weighted_history.detach().float()
            sequence = torch.cat([weighted_history, current_group], dim=1)
            sequence = sequence.permute(0, 2, 1, 3).reshape(
                batch_indices.numel() * num_modes, -1, channels
            )
            self.last_group_records.append(
                {
                    "valid_len": int(valid_len),
                    "num_samples": int(batch_indices.numel()),
                    "history_group_shape": list(history_group.shape),
                    "history_beta_shape": beta_shape,
                    "history_beta_stats": beta_stats,
                    "history_weight_source": history_weight_source,
                    "history_weight_shape": history_weight_shape,
                    "history_weight_stats": history_weight_stats,
                    "score_modulation_order": (
                        "history_query * sigmoid(history_score) * w_i"
                        if effective_history_weight is not None
                        else "history_query * sigmoid(history_score)"
                    ),
                    "score_sigmoid_stats": {
                        "mean": float(score_sigmoid_detached.mean().cpu().item()),
                        "std": float(score_sigmoid_detached.std(unbiased=False).cpu().item()),
                        "min": float(score_sigmoid_detached.min().cpu().item()),
                        "max": float(score_sigmoid_detached.max().cpu().item()),
                    },
                    "score_weighted_history_norm": float(
                        score_weighted_history_detached.norm().cpu().item()
                    ),
                    "weighted_history_norm": float(
                        weighted_history_detached.norm().cpu().item()
                    ),
                    "weighted_history_reconstruction_error_max": reconstruction_error,
                    "sequence_shape": list(sequence.shape),
                    "input_order": "oldest_to_newest_then_current",
                    "final_modulation_slot_records": final_modulation_records,
                    "low_score_threshold": low_score_threshold,
                    "stale_w_gt_1_count": int(stale_w_gt_1_count),
                    "low_score_stale_w_gt_1_count": int(
                        low_score_stale_w_gt_1_count
                    ),
                    "allocation_audit_attached": allocation_audit is not None,
                }
            )

            modeled = self.sequence_model(sequence)
            delta = modeled[:, -1, :].reshape(
                batch_indices.numel(), num_modes, channels
            ).unsqueeze(1)
            delta = self.output_norm(delta)
            group_output = current_group + self.alpha.to(
                dtype=current_query.dtype
            ) * delta
            output = output.index_copy(0, batch_indices, group_output)
        self._debug_print(
            "sequence_model used "
            f"valid_count_values={valid_counts.detach().cpu().tolist()} "
            f"output_shape={tuple(output.shape)} "
            f"output_requires_grad={output.requires_grad} "
            f"output_finite={bool(torch.isfinite(output).all().item())}"
        )

        return output


def _run_shape_test() -> None:
    batch_size, history_steps, num_modes, embed_dims = 2, 2, 18, 256
    device = torch.device(
        "cuda" if RealMamba is not None and torch.cuda.is_available() else "cpu"
    )
    current_query = torch.randn(
        batch_size, 1, num_modes, embed_dims, dtype=torch.float32, device=device
    )
    history_query = torch.randn(
        batch_size, history_steps, num_modes, embed_dims,
        dtype=torch.float32, device=device
    )
    history_score = torch.randn(
        batch_size, history_steps, num_modes, dtype=torch.float32, device=device
    )
    history_valid_mask = torch.tensor(
        [[False, True], [True, True]], device=device
    )

    smm = SelectiveMomentumMemory(
        embed_dims=embed_dims,
        num_modes=num_modes,
        history_len=history_steps,
        use_real_mamba=True,
        fallback_if_unavailable=True,
    ).to(device)
    output = smm(
        current_query=current_query,
        history_query=history_query,
        history_score=history_score,
        history_valid_mask=history_valid_mask,
    )

    assert output.shape == (2, 1, 18, 256)
    assert output.dtype == current_query.dtype
    assert not torch.isnan(output).any()
    assert torch.isfinite(output).all()

    padded_smm = SelectiveMomentumMemory(
        embed_dims=embed_dims,
        num_modes=num_modes,
        history_len=history_steps,
        use_real_mamba=True,
        fallback_if_unavailable=True,
    ).to(device)
    compact_smm = SelectiveMomentumMemory(
        embed_dims=embed_dims,
        num_modes=num_modes,
        history_len=history_steps,
        use_real_mamba=True,
        fallback_if_unavailable=True,
    ).to(device)
    compact_smm.load_state_dict(padded_smm.state_dict())
    one_history = torch.randn(1, 1, num_modes, embed_dims, device=device)
    one_score = torch.randn(1, 1, num_modes, device=device)
    one_current = torch.randn(1, 1, num_modes, embed_dims, device=device)
    zero_history = torch.zeros_like(one_history)
    zero_score = torch.zeros_like(one_score)
    padded_output = padded_smm(
        current_query=one_current,
        history_query=torch.cat([zero_history, one_history], dim=1),
        history_score=torch.cat([zero_score, one_score], dim=1),
        history_valid_mask=torch.tensor([[False, True]], device=device),
    )
    compact_output = compact_smm(
        current_query=one_current,
        history_query=one_history,
        history_score=one_score,
        history_valid_mask=torch.tensor([[True]], device=device),
    )
    max_abs_diff = (padded_output - compact_output).abs().max().item()
    assert max_abs_diff == 0.0, max_abs_diff

    backend = "real_mamba" if smm.using_real_mamba else "fallback"
    print(
        "SelectiveMomentumMemory shape test passed "
        f"with backend={backend}, device={device}, "
        f"padded_vs_compact_max_abs_diff={max_abs_diff}"
    )


if __name__ == "__main__":
    _run_shape_test()
