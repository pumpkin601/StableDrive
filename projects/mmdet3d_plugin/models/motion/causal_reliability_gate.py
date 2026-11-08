from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


CALIBRATION_FEATURE_NAMES = ("e_pos", "e_vel", "e_yaw", "e_progress")
RELIABILITY_FEATURE_NAMES = (
    "e_pos",
    "e_vel",
    "e_yaw",
    "e_progress",
    "current_speed",
    "age_seconds",
)
CALIBRATION_ERROR_BINS = {
    "e_pos": [0.0, 0.5, 1.0, 2.0, 5.0, float("inf")],
    "e_vel": [0.0, 0.5, 1.0, 2.0, 5.0, float("inf")],
    "e_yaw": [0.0, 0.1, 0.3, 0.7, float("inf")],
    "e_progress": [0.0, 0.5, 1.0, 2.0, 5.0, float("inf")],
}


class StaleEvidenceReleaseMLP(nn.Module):
    """Bounded release score for CR-v1.2 stale-history penalty."""

    def __init__(
        self,
        input_dims: int,
        hidden_dims: int = 64,
        release_init_mode: str = "neutral",
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(input_dims), int(hidden_dims)),
            nn.ReLU(inplace=True),
            nn.LayerNorm(int(hidden_dims)),
            nn.Linear(int(hidden_dims), 1),
        )
        self.release_init_mode = str(release_init_mode)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        final = self.net[-1]
        nn.init.zeros_(final.weight)
        if self.release_init_mode == "neutral":
            final.bias.data.zero_()
        elif self.release_init_mode == "closed":
            final.bias.data.fill_(-4.0)
        elif self.release_init_mode == "open":
            final.bias.data.fill_(4.0)
        else:
            raise ValueError(
                "release_init_mode must be one of {'neutral', 'closed', 'open'}"
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(features).squeeze(-1))


class RetrospectiveReliabilityGate(nn.Module):
    """Frame-level causal reliability gate for CR-v1.

    Normalization uses fixed physical scales only. Validation/test statistics
    must never be used to fit these constants.
    """

    def __init__(
        self,
        embed_dims: int = 256,
        hidden_dims: int = 64,
        relation_dims: int = 16,
        age_embed_dims: int = 8,
        beta_init: float = 0.9,
        clamp_max: float = 5.0,
        detach_relation_context: bool = True,
        weight_mode: str = "independent_sigmoid",
        softmax_temperature: float = 1.0,
        cr_v1_2_enabled: bool = False,
        stale_age_threshold: int = 3,
        kappa_age_t3: float = 0.0,
        kappa_age_t4: float = 0.0,
        release_hidden_dims: Optional[int] = None,
        release_init_mode: str = "neutral",
    ) -> None:
        super().__init__()
        if not (0.0 < float(beta_init) < 1.0):
            raise ValueError("beta_init must be in (0, 1)")
        if weight_mode not in ("independent_sigmoid", "competitive_softmax"):
            raise ValueError(
                "weight_mode must be one of {'independent_sigmoid', "
                "'competitive_softmax'}"
            )
        if float(softmax_temperature) <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        self.embed_dims = int(embed_dims)
        self.hidden_dims = int(hidden_dims)
        self.relation_dims = int(relation_dims)
        self.age_embed_dims = int(age_embed_dims)
        self.beta_init = float(beta_init)
        self.clamp_max = float(clamp_max)
        self.detach_relation_context = bool(detach_relation_context)
        self.weight_mode = str(weight_mode)
        self.softmax_temperature = float(softmax_temperature)
        self.cr_v1_2_enabled = bool(cr_v1_2_enabled)
        self.stale_age_threshold = int(stale_age_threshold)
        self.release_init_mode = str(release_init_mode)
        if self.cr_v1_2_enabled and self.weight_mode != "competitive_softmax":
            raise ValueError("CR-v1.2 requires weight_mode='competitive_softmax'")
        if self.stale_age_threshold < 1 or self.stale_age_threshold > 4:
            raise ValueError("stale_age_threshold must be in [1, 4]")

        self.current_proj = nn.Linear(self.embed_dims, self.relation_dims)
        self.history_proj = nn.Linear(self.embed_dims, self.relation_dims)
        self.age_embedding = nn.Embedding(5, self.age_embed_dims)
        relation_feature_dims = 1 + 2 * self.relation_dims
        input_dims = 4 + 4 + 1 + 1 + self.age_embed_dims + relation_feature_dims
        self._mlp_input_dims = int(input_dims)
        self.mlp = nn.Sequential(
            nn.Linear(input_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.LayerNorm(self.hidden_dims),
            nn.Linear(self.hidden_dims, self.hidden_dims),
            nn.ReLU(inplace=True),
            nn.Linear(self.hidden_dims, 1),
        )
        self.register_buffer(
            "calibration_scales",
            torch.tensor([10.0, 10.0, math.pi, 10.0], dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "current_speed_scale",
            torch.tensor(20.0, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "age_seconds_scale",
            torch.tensor(2.0, dtype=torch.float32),
            persistent=False,
        )
        kappa_by_age = torch.zeros(5, dtype=torch.float32)
        kappa_by_age[3] = float(kappa_age_t3)
        kappa_by_age[4] = float(kappa_age_t4)
        self.register_buffer(
            "cr_v1_2_kappa_by_age",
            kappa_by_age,
            persistent=True,
        )
        self.stale_release_mlp = None
        if self.cr_v1_2_enabled:
            self.stale_release_mlp = StaleEvidenceReleaseMLP(
                input_dims=input_dims,
                hidden_dims=int(release_hidden_dims or hidden_dims),
                release_init_mode=self.release_init_mode,
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        final = self.mlp[-1]
        nn.init.zeros_(final.weight)
        if self.weight_mode == "competitive_softmax":
            # Equal logits produce w_i = 1 over valid histories after
            # valid-count scaling, matching C2 at initialization while still
            # forcing future learning to be relative across frames.
            final.bias.data.zero_()
        else:
            final.bias.data.fill_(math.log(self.beta_init / (1.0 - self.beta_init)))
        if self.stale_release_mlp is not None:
            self.stale_release_mlp.reset_parameters()

    def _zero_release_dependency(self, reference: torch.Tensor) -> torch.Tensor:
        if self.stale_release_mlp is None:
            return reference.new_zeros(())
        zero = reference.new_zeros(())
        for param in self.stale_release_mlp.parameters():
            zero = zero + param.to(dtype=reference.dtype).sum() * 0.0
        return zero

    def _normalize_features(
        self,
        reliability_features: torch.Tensor,
        reliability_availability: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        calibration = reliability_features[..., :4].abs()
        availability = reliability_availability[..., :4].to(
            device=reliability_features.device,
            dtype=reliability_features.dtype,
        )
        scales = self.calibration_scales.to(
            device=reliability_features.device,
            dtype=reliability_features.dtype,
        )
        calibration_norm = torch.clamp(calibration / scales, 0.0, self.clamp_max)
        calibration_norm = calibration_norm * availability
        current_speed_norm = torch.clamp(
            reliability_features[..., 4:5].abs()
            / self.current_speed_scale.to(
                device=reliability_features.device,
                dtype=reliability_features.dtype,
            ),
            0.0,
            self.clamp_max,
        )
        age_seconds = reliability_features[..., 5:6].abs()
        age_seconds_norm = torch.clamp(
            age_seconds
            / self.age_seconds_scale.to(
                device=reliability_features.device,
                dtype=reliability_features.dtype,
            ),
            0.0,
            self.clamp_max,
        )
        return calibration_norm, availability, current_speed_norm, age_seconds_norm

    def _history_age_index(self, reliability_features: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            torch.round(reliability_features[..., 5] / 0.5).long(), min=0, max=4
        )

    def _cr_v1_2_penalty_tensors(
        self,
        base_logits: torch.Tensor,
        reliability_features: torch.Tensor,
        valid_bool: torch.Tensor,
        stale_release_score: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        age_index = self._history_age_index(reliability_features).to(
            device=base_logits.device
        )
        stale_mask = (age_index >= self.stale_age_threshold) & valid_bool
        kappa = self.cr_v1_2_kappa_by_age.to(
            device=base_logits.device, dtype=base_logits.dtype
        )[age_index]
        release = stale_release_score.to(device=base_logits.device, dtype=base_logits.dtype)
        effective_penalty = (
            stale_mask.to(dtype=base_logits.dtype)
            * kappa
            * (1.0 - release).clamp(0.0, 1.0)
        )
        recency_safe_logit = base_logits - effective_penalty
        return {
            "age_index": age_index,
            "stale_mask": stale_mask,
            "kappa_age": kappa,
            "stale_release_score": release,
            "effective_stale_penalty": effective_penalty,
            "recency_safe_logit": recency_safe_logit,
        }

    def _cr_v1_2_audit_payload(
        self,
        reliability_features: torch.Tensor,
        reliability_availability: torch.Tensor,
        valid_bool: torch.Tensor,
        base_logits: torch.Tensor,
        penalty_tensors: Dict[str, torch.Tensor],
        allocation_weight: torch.Tensor,
    ) -> Dict[str, object]:
        base_detached = base_logits.detach().float()
        weight_detached = allocation_weight.detach().float()
        age_index = penalty_tensors["age_index"].detach()
        stale_mask = penalty_tensors["stale_mask"].detach()
        kappa = penalty_tensors["kappa_age"].detach().float()
        release = penalty_tensors["stale_release_score"].detach().float()
        penalty = penalty_tensors["effective_stale_penalty"].detach().float()
        z = penalty_tensors["recency_safe_logit"].detach().float()
        evidence = reliability_features.detach().float()
        availability = reliability_availability.detach().float()
        slot_records = []
        sample_records = []
        stale_valid_release = release[stale_mask]
        stale_valid_penalty = penalty[stale_mask]
        for batch_idx in range(weight_detached.shape[0]):
            sample_mask = valid_bool[batch_idx]
            valid_weights = weight_detached[batch_idx][sample_mask]
            valid_ages = age_index[batch_idx][sample_mask]
            valid_stale = stale_mask[batch_idx][sample_mask]
            valid_penalty = penalty[batch_idx][sample_mask]
            valid_release = release[batch_idx][sample_mask]
            valid_count = int(valid_weights.numel())
            sum_valid = float(valid_weights.sum().cpu().item()) if valid_count else 0.0
            expected = float(valid_count)
            if valid_count > 1:
                probs = valid_weights / valid_weights.sum().clamp_min(1e-12)
                entropy = float(
                    (-(probs * torch.log(probs.clamp_min(1e-12))).sum()
                    / math.log(float(valid_count))).cpu().item()
                )
                weight_range = float(
                    (valid_weights.max() - valid_weights.min()).cpu().item()
                )
            else:
                entropy = None
                weight_range = 0.0 if valid_count == 1 else None
            max_weight_age = None
            if valid_count:
                max_pos = int(torch.argmax(valid_weights).cpu().item())
                max_weight_age = int(valid_ages[max_pos].cpu().item())
            sample_records.append(
                {
                    "sample_index": int(batch_idx),
                    "L_eff": valid_count,
                    "sum_valid_w_minus_L_eff": float(sum_valid - expected),
                    "entropy": entropy,
                    "weight_range": weight_range,
                    "max_weight_age": max_weight_age,
                    "stale_w_gt_1_count": int(
                        ((valid_weights > 1.0) & valid_stale).sum().cpu().item()
                    ),
                    "stale_release_mean": float(
                        valid_release[valid_stale].mean().cpu().item()
                    )
                    if bool(valid_stale.any().cpu().item())
                    else None,
                    "stale_penalty_mean": float(
                        valid_penalty[valid_stale].mean().cpu().item()
                    )
                    if bool(valid_stale.any().cpu().item())
                    else None,
                }
            )
            for slot_idx in range(weight_detached.shape[1]):
                slot_records.append(
                    {
                        "sample_index": int(batch_idx),
                        "slot_index": int(slot_idx),
                        "age_i": int(age_index[batch_idx, slot_idx].cpu().item()),
                        "valid_mask_i": bool(valid_bool[batch_idx, slot_idx].cpu().item()),
                        "base_competitive_logit_g_i": float(
                            base_detached[batch_idx, slot_idx].cpu().item()
                        ),
                        "stale_mask_i": bool(stale_mask[batch_idx, slot_idx].cpu().item()),
                        "kappa_age_i": float(kappa[batch_idx, slot_idx].cpu().item()),
                        "causal_evidence_feature_summary": {
                            name: float(
                                evidence[batch_idx, slot_idx, feat_idx].cpu().item()
                            )
                            for feat_idx, name in enumerate(RELIABILITY_FEATURE_NAMES)
                        },
                        "causal_evidence_availability": {
                            name: float(
                                availability[batch_idx, slot_idx, feat_idx].cpu().item()
                            )
                            for feat_idx, name in enumerate(CALIBRATION_FEATURE_NAMES)
                        },
                        "stale_release_score_q_i": float(
                            release[batch_idx, slot_idx].cpu().item()
                        ),
                        "effective_stale_penalty_i": float(
                            penalty[batch_idx, slot_idx].cpu().item()
                        ),
                        "recency_safe_logit_z_i": float(
                            z[batch_idx, slot_idx].cpu().item()
                        ),
                        "competitive_weight_w_i": float(
                            weight_detached[batch_idx, slot_idx].cpu().item()
                        ),
                    }
                )
        return {
            "enabled": True,
            "stale_age_threshold": int(self.stale_age_threshold),
            "kappa_age_t3": float(self.cr_v1_2_kappa_by_age[3].detach().cpu().item()),
            "kappa_age_t4": float(self.cr_v1_2_kappa_by_age[4].detach().cpu().item()),
            "release_forward_count": 1,
            "stale_release_finite": bool(torch.isfinite(release).all().cpu().item()),
            "stale_release_mean": float(release[valid_bool].mean().cpu().item())
            if bool(valid_bool.any().cpu().item())
            else None,
            "stale_release_min": float(release[valid_bool].min().cpu().item())
            if bool(valid_bool.any().cpu().item())
            else None,
            "stale_release_max": float(release[valid_bool].max().cpu().item())
            if bool(valid_bool.any().cpu().item())
            else None,
            "stale_release_lt_0_05_fraction": float(
                (stale_valid_release < 0.05).float().mean().cpu().item()
            )
            if stale_valid_release.numel()
            else None,
            "stale_release_gt_0_95_fraction": float(
                (stale_valid_release > 0.95).float().mean().cpu().item()
            )
            if stale_valid_release.numel()
            else None,
            "effective_stale_penalty_min": float(
                stale_valid_penalty.min().cpu().item()
            )
            if stale_valid_penalty.numel()
            else None,
            "effective_stale_penalty_max": float(
                stale_valid_penalty.max().cpu().item()
            )
            if stale_valid_penalty.numel()
            else None,
            "z_minus_g_max": float((z - base_detached).max().cpu().item()),
            "z_minus_g_min": float((z - base_detached).min().cpu().item()),
            "slot_records": slot_records,
            "sample_records": sample_records,
        }

    def forward(
        self,
        current_query: torch.Tensor,
        history_query: torch.Tensor,
        reliability_features: torch.Tensor,
        reliability_availability: torch.Tensor,
        history_valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, object]]:
        if current_query.dim() == 4:
            current_context = current_query.squeeze(1).mean(dim=1)
        else:
            raise ValueError(f"Expected current_query [B,1,M,D], got {tuple(current_query.shape)}")
        if history_query.dim() != 4:
            raise ValueError(f"Expected history_query [B,H,M,D], got {tuple(history_query.shape)}")
        if reliability_features.shape[:2] != history_query.shape[:2]:
            raise ValueError("reliability_features must match [B,H]")
        if reliability_features.shape[-1] != len(RELIABILITY_FEATURE_NAMES):
            raise ValueError(
                f"Expected {len(RELIABILITY_FEATURE_NAMES)} reliability features, "
                f"got {reliability_features.shape[-1]}"
            )
        if reliability_availability.shape[:2] != history_query.shape[:2] or reliability_availability.shape[-1] != 4:
            raise ValueError("reliability_availability must have shape [B,H,4]")
        if history_valid_mask.shape != history_query.shape[:2]:
            raise ValueError("history_valid_mask must have shape [B,H]")

        history_context = history_query.mean(dim=2)
        if self.detach_relation_context:
            current_context = current_context.detach()
            history_context = history_context.detach()

        calibration_norm, availability, speed_norm, age_norm = self._normalize_features(
            reliability_features, reliability_availability
        )
        age_index = self._history_age_index(reliability_features)
        age_emb = self.age_embedding(age_index)
        current_expanded = current_context[:, None, :].expand_as(history_context)
        cosine = F.cosine_similarity(current_expanded, history_context, dim=-1).unsqueeze(-1)
        cosine = torch.clamp(cosine, -1.0, 1.0)
        current_proj = self.current_proj(current_expanded)
        history_proj = self.history_proj(history_context)
        relation = torch.cat(
            [
                cosine,
                torch.clamp((current_proj - history_proj).abs(), 0.0, self.clamp_max),
                torch.clamp(current_proj * history_proj, -1.0, 1.0),
            ],
            dim=-1,
        )
        mlp_input = torch.cat(
            [calibration_norm, availability, speed_norm, age_norm, age_emb, relation],
            dim=-1,
        )
        base_logits = self.mlp(mlp_input).squeeze(-1)
        valid_bool = history_valid_mask.to(device=base_logits.device, dtype=torch.bool)
        valid = valid_bool.to(dtype=base_logits.dtype)
        logits = base_logits
        cr_v1_2_audit = None
        if self.cr_v1_2_enabled:
            if self.stale_release_mlp is None:
                raise RuntimeError("CR-v1.2 enabled but release module is missing")
            stale_release_score = self.stale_release_mlp(mlp_input.detach())
            penalty_tensors = self._cr_v1_2_penalty_tensors(
                base_logits=base_logits,
                reliability_features=reliability_features,
                valid_bool=valid_bool,
                stale_release_score=stale_release_score,
            )
            logits = penalty_tensors["recency_safe_logit"]
        if self.weight_mode == "competitive_softmax":
            masked_logits = logits.masked_fill(~valid_bool, -1.0e4)
            probs = torch.softmax(masked_logits / self.softmax_temperature, dim=1)
            probs = probs * valid
            valid_counts = valid.sum(dim=1, keepdim=True)
            allocation_weight = probs * valid_counts.clamp_min(1.0)
            allocation_weight = allocation_weight * valid
        else:
            allocation_weight = torch.sigmoid(logits)
            allocation_weight = allocation_weight * valid
        if self.cr_v1_2_enabled:
            allocation_weight = allocation_weight + self._zero_release_dependency(
                allocation_weight
            )
            cr_v1_2_audit = self._cr_v1_2_audit_payload(
                reliability_features=reliability_features,
                reliability_availability=reliability_availability,
                valid_bool=valid_bool,
                base_logits=base_logits,
                penalty_tensors=penalty_tensors,
                allocation_weight=allocation_weight,
            )
        finite = bool(torch.isfinite(allocation_weight).all().detach().cpu().item())
        valid_allocation_weight = allocation_weight[valid_bool]
        age_seconds = reliability_features[..., 5].detach()
        weight_detached = allocation_weight.detach()
        logits_detached = logits.detach()
        per_sample_ranges = []
        allocation_entropies = []
        weight_sum_errors = []
        allocation_invariant_records = []
        rank_agreement_count = 0
        rank_agreement_total = 0
        for batch_idx in range(weight_detached.shape[0]):
            sample_mask = valid_bool[batch_idx]
            vals = weight_detached[batch_idx][sample_mask].float()
            sample_logits = logits_detached[batch_idx][sample_mask].float()
            invalid_vals = weight_detached[batch_idx][~sample_mask].float()
            expected_sum = float(vals.numel())
            valid_weight_sum = float(vals.sum().cpu().item()) if vals.numel() else 0.0
            invalid_weight_sum = (
                float(invalid_vals.abs().sum().cpu().item())
                if invalid_vals.numel()
                else 0.0
            )
            sum_error = abs(valid_weight_sum - expected_sum)
            allocation_invariant_records.append(
                {
                    "sample_index": int(batch_idx),
                    "L_eff": int(vals.numel()),
                    "valid_weight_sum": valid_weight_sum,
                    "valid_weight_sum_abs_error": sum_error,
                    "invalid_weight_sum": invalid_weight_sum,
                    "current_query_allocation_participation": 0,
                    "fp32_invariant_checked": True,
                    "valid_weight_min": float(vals.min().cpu().item())
                    if vals.numel()
                    else None,
                    "valid_weight_max": float(vals.max().cpu().item())
                    if vals.numel()
                    else None,
                    "valid_weight_range": float((vals.max() - vals.min()).cpu().item())
                    if vals.numel()
                    else None,
                }
            )
            if vals.numel() == 0:
                continue
            weight_sum_errors.append(sum_error)
            if vals.numel() > 1:
                per_sample_ranges.append(float((vals.max() - vals.min()).cpu().item()))
                prob = vals / vals.sum().clamp_min(1e-12)
                entropy = -(prob * torch.log(prob.clamp_min(1e-12))).sum()
                norm_entropy = entropy / math.log(float(vals.numel()))
                allocation_entropies.append(float(norm_entropy.cpu().item()))
                rank_agreement_count += int(
                    torch.argmax(vals).item() == torch.argmax(sample_logits).item()
                )
                rank_agreement_total += 1
        by_age = {}
        age_means_for_variance = []
        for age in range(1, 5):
            mask = valid_bool & (torch.round(age_seconds / 0.5).long() == age)
            vals = weight_detached[mask]
            if vals.numel():
                age_means_for_variance.append(vals.float().mean())
            by_age[f"t-{age}"] = {
                "count": int(vals.numel()),
                "mean": float(vals.float().mean().cpu().item()) if vals.numel() else None,
                "std": float(vals.float().std(unbiased=False).cpu().item()) if vals.numel() else None,
                "min": float(vals.float().min().cpu().item()) if vals.numel() else None,
                "max": float(vals.float().max().cpu().item()) if vals.numel() else None,
            }
        age_mean_tensor = (
            torch.stack(age_means_for_variance)
            if age_means_for_variance
            else torch.empty(0, device=base_logits.device)
        )
        by_error_bin = {}
        raw_calibration = reliability_features[..., :4].detach().abs()
        availability_bool = reliability_availability[..., :4].to(
            device=base_logits.device, dtype=torch.bool
        )
        for feat_idx, feat_name in enumerate(CALIBRATION_FEATURE_NAMES):
            bins = CALIBRATION_ERROR_BINS[feat_name]
            feat_bins = {}
            raw = raw_calibration[..., feat_idx]
            for bin_idx in range(len(bins) - 1):
                lo, hi = bins[bin_idx], bins[bin_idx + 1]
                if math.isinf(hi):
                    mask = valid_bool & availability_bool[..., feat_idx] & (raw >= lo)
                    label = f"[{lo},inf)"
                else:
                    mask = (
                        valid_bool
                        & availability_bool[..., feat_idx]
                        & (raw >= lo)
                        & (raw < hi)
                    )
                    label = f"[{lo},{hi})"
                vals = weight_detached[mask]
                feat_bins[label] = {
                    "count": int(vals.numel()),
                    "mean": float(vals.float().mean().cpu().item()) if vals.numel() else None,
                }
            by_error_bin[feat_name] = feat_bins
        stats = {
            "feature_names": list(RELIABILITY_FEATURE_NAMES),
            "calibration_feature_names": list(CALIBRATION_FEATURE_NAMES),
            "normalization": {
                "e_pos": "fixed_physical_scale_10m",
                "e_vel": "fixed_physical_scale_10m_per_s",
                "e_yaw": "fixed_physical_scale_pi",
                "e_progress": "fixed_physical_scale_10m",
                "current_speed": "fixed_physical_scale_20m_per_s",
                "age_seconds": "fixed_physical_scale_2s",
                "source": "fixed_physical_scales_not_val_or_test_statistics",
            },
            "availability_mask_enters_mlp": True,
            "weight_mode": self.weight_mode,
            "competitive_allocation": bool(self.weight_mode == "competitive_softmax"),
            "softmax_temperature": self.softmax_temperature
            if self.weight_mode == "competitive_softmax"
            else None,
            "allocation_weight_shape": list(allocation_weight.shape),
            "allocation_weight_finite": finite,
            "valid_allocation_weight_count": int(valid_allocation_weight.numel()),
            "allocation_weight_mean": float(valid_allocation_weight.detach().float().mean().cpu().item())
            if valid_allocation_weight.numel() > 0
            else None,
            "allocation_weight_std": float(valid_allocation_weight.detach().float().std(unbiased=False).cpu().item())
            if valid_allocation_weight.numel() > 0
            else None,
            "allocation_weight_variance": float(valid_allocation_weight.detach().float().var(unbiased=False).cpu().item())
            if valid_allocation_weight.numel() > 0
            else None,
            "allocation_weight_age_variance": float(age_mean_tensor.detach().float().var(unbiased=False).cpu().item())
            if age_mean_tensor.numel() > 1
            else 0.0 if age_mean_tensor.numel() == 1 else None,
            "allocation_weight_min": float(valid_allocation_weight.detach().float().min().cpu().item())
            if valid_allocation_weight.numel() > 0
            else None,
            "allocation_weight_max": float(valid_allocation_weight.detach().float().max().cpu().item())
            if valid_allocation_weight.numel() > 0
            else None,
            "fraction_allocation_weight_lt_0_05": float((valid_allocation_weight < 0.05).float().mean().cpu().item())
            if valid_allocation_weight.numel() > 0
            else None,
            "fraction_allocation_weight_gt_0_95": float((valid_allocation_weight > 0.95).float().mean().cpu().item())
            if valid_allocation_weight.numel() > 0
            else None,
            "fraction_allocation_weight_gt_0_995": float((valid_allocation_weight > 0.995).float().mean().cpu().item())
            if valid_allocation_weight.numel() > 0
            else None,
            "per_sample_weight_range_mean": sum(per_sample_ranges) / len(per_sample_ranges)
            if per_sample_ranges
            else None,
            "per_sample_weight_range_max": max(per_sample_ranges)
            if per_sample_ranges
            else None,
            "allocation_entropy_mean": sum(allocation_entropies) / len(allocation_entropies)
            if allocation_entropies
            else None,
            "allocation_entropy_min": min(allocation_entropies)
            if allocation_entropies
            else None,
            "valid_weight_sum_error_max": max(weight_sum_errors)
            if weight_sum_errors
            else None,
            "allocation_invariant_records": allocation_invariant_records,
            "logit_weight_argmax_agreement": (
                rank_agreement_count / rank_agreement_total
                if rank_agreement_total
                else None
            ),
            "allocation_weight_by_age": by_age,
            "allocation_weight_by_calibration_error_bin": by_error_bin,
        }
        if cr_v1_2_audit is not None:
            stats["cr_v1_2_allocation_audit"] = cr_v1_2_audit
            stats["allocation_policy"] = "cr_v1_2_evidence_gated_recency_safe"
        else:
            stats.update(
                {
                    "beta_shape": list(allocation_weight.shape),
                    "beta_finite": finite,
                    "valid_beta_count": int(valid_allocation_weight.numel()),
                    "beta_mean": stats["allocation_weight_mean"],
                    "beta_std": stats["allocation_weight_std"],
                    "beta_variance": stats["allocation_weight_variance"],
                    "beta_age_variance": stats["allocation_weight_age_variance"],
                    "beta_min": stats["allocation_weight_min"],
                    "beta_max": stats["allocation_weight_max"],
                    "fraction_beta_lt_0_05": stats["fraction_allocation_weight_lt_0_05"],
                    "fraction_beta_gt_0_95": stats["fraction_allocation_weight_gt_0_95"],
                    "fraction_beta_gt_0_995": stats["fraction_allocation_weight_gt_0_995"],
                    "beta_by_age": by_age,
                    "beta_by_calibration_error_bin": by_error_bin,
                }
            )
        return allocation_weight, stats
