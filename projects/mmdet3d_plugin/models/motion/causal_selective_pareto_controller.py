"""Causal selective Pareto controller for frozen planning candidates.

Unlike unrestricted logit calibration, CSPC keeps the frozen base-selected
mode unless one alternative passes a learned takeover test and deterministic
multi-horizon guards.  All proposer and scene inputs are detached, so scorer
losses cannot corrupt candidate geometry or shared perception features.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .causal_dense_risk_scorer import (
    CausalDenseRiskScorer,
    COLLISION_INDICES,
    CONSISTENCY_INDICES,
    L2_INDICES,
)


class CausalSelectiveParetoController(CausalDenseRiskScorer):
    """Frozen-proposer dense critic with conservative one-mode takeover."""

    def __init__(
        self,
        takeover_probability_threshold: float = 0.90,
        takeover_logit_epsilon: float = 1e-3,
        predicted_l2_ratio_max: float = 1.0,
        predicted_collision_delta_max: float = 0.02,
        predicted_consistency_ratio_max: float = 1.01,
        predicted_weighted_ratio_max: float = 0.98,
        takeover_loss_weight: float = 1.0,
        takeover_positive_weight: float = 5.0,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.takeover_probability_threshold = float(
            takeover_probability_threshold
        )
        self.takeover_logit_epsilon = float(takeover_logit_epsilon)
        self.predicted_l2_ratio_max = float(predicted_l2_ratio_max)
        self.predicted_collision_delta_max = float(
            predicted_collision_delta_max
        )
        self.predicted_consistency_ratio_max = float(
            predicted_consistency_ratio_max
        )
        self.predicted_weighted_ratio_max = float(
            predicted_weighted_ratio_max
        )
        self.takeover_loss_weight = float(takeover_loss_weight)
        self.takeover_positive_weight = float(takeover_positive_weight)
        if not 0.5 < self.takeover_probability_threshold < 1.0:
            raise ValueError("CSPC takeover probability threshold is invalid")
        if self.takeover_logit_epsilon <= 0.0:
            raise ValueError("CSPC takeover epsilon must be positive")
        if self.predicted_l2_ratio_max > 1.0:
            raise ValueError("CSPC may not permit predicted L2 regression")
        if self.predicted_collision_delta_max < 0.0:
            raise ValueError("CSPC collision tolerance must be non-negative")
        if self.predicted_consistency_ratio_max > 1.01:
            raise ValueError("CSPC consistency tolerance exceeds the lock")
        if self.predicted_weighted_ratio_max > 0.98:
            raise ValueError("CSPC weighted guard exceeds the lock")
        if self.takeover_loss_weight <= 0.0 or self.takeover_positive_weight < 1.0:
            raise ValueError("CSPC takeover loss settings are invalid")

        relative_dims = 2 * self.token_dims + len(self.metric_scales) + 1
        self.takeover_head = nn.Sequential(
            nn.LayerNorm(relative_dims),
            nn.Linear(relative_dims, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, self.token_dims // 2),
            nn.GELU(),
            nn.Linear(self.token_dims // 2, 1),
        )
        nn.init.zeros_(self.takeover_head[-1].weight)
        nn.init.constant_(
            self.takeover_head[-1].bias,
            math.log(0.05 / 0.95),
        )

    @staticmethod
    def _gather_mode(value: torch.Tensor, mode: torch.Tensor) -> torch.Tensor:
        gather_index = mode
        while gather_index.dim() < value.dim():
            gather_index = gather_index.unsqueeze(-1)
        shape = list(value.shape)
        shape[2] = 1
        gather_index = gather_index.expand(*shape)
        return value.gather(2, gather_index).squeeze(2)

    @staticmethod
    def _safe_ratio(candidate: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
        regular = candidate / baseline.clamp_min(1e-6)
        zero_case = torch.where(
            candidate <= baseline + 1e-8,
            torch.ones_like(candidate),
            torch.full_like(candidate, 1e6),
        )
        return torch.where(baseline.abs() > 1e-6, regular, zero_case)

    def _relative_features(
        self,
        candidate_features: torch.Tensor,
        score_metrics: torch.Tensor,
        base_logits: torch.Tensor,
        base_mode: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = candidate_features.shape[0]
        features = candidate_features.reshape(
            batch_size, self.num_commands, self.num_modes, self.token_dims
        )
        metrics = score_metrics.reshape(
            batch_size,
            self.num_commands,
            self.num_modes,
            len(self.metric_scales),
        )
        base_feature = self._gather_mode(features, base_mode)
        base_metric = self._gather_mode(metrics, base_mode)
        base_logit = self._gather_mode(base_logits, base_mode)
        return torch.cat(
            [
                features,
                features - base_feature.unsqueeze(2),
                metrics - base_metric.unsqueeze(2),
                (base_logits - base_logit.unsqueeze(2)).unsqueeze(-1),
            ],
            dim=-1,
        )

    def _predicted_guard(
        self,
        score_metrics: torch.Tensor,
        base_mode: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        batch_size = score_metrics.shape[0]
        metrics = score_metrics.reshape(
            batch_size,
            self.num_commands,
            self.num_modes,
            len(self.metric_scales),
        )
        baseline = self._gather_mode(metrics, base_mode).unsqueeze(2)
        l2 = metrics[..., list(L2_INDICES)]
        base_l2 = baseline[..., list(L2_INDICES)]
        collision = metrics[..., list(COLLISION_INDICES)]
        base_collision = baseline[..., list(COLLISION_INDICES)]
        consistency = metrics[..., list(CONSISTENCY_INDICES)]
        base_consistency = baseline[..., list(CONSISTENCY_INDICES)]
        l2_guard = (
            self._safe_ratio(l2, base_l2) <= self.predicted_l2_ratio_max
        ).all(dim=-1)
        collision_guard = (
            collision <= base_collision + self.predicted_collision_delta_max
        ).all(dim=-1)
        consistency_guard = (
            self._safe_ratio(consistency, base_consistency)
            <= self.predicted_consistency_ratio_max
        ).all(dim=-1)
        weights = self.teacher_metric_weights.to(
            device=metrics.device, dtype=metrics.dtype
        ).view(1, 1, 1, -1)
        weighted_ratio = (
            self._safe_ratio(metrics, baseline) * weights
        ).sum(dim=-1)
        weighted_guard = weighted_ratio <= self.predicted_weighted_ratio_max
        return {
            "all": l2_guard & collision_guard & consistency_guard & weighted_guard,
            "l2": l2_guard,
            "collision": collision_guard,
            "consistency": consistency_guard,
            "weighted": weighted_guard,
            "weighted_ratio": weighted_ratio,
        }

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
        # The soft auxiliary critic is deliberately isolated from the proposer.
        soft_output = super().forward(
            base_logits=base_logits.detach(),
            plan_reg=plan_reg.detach(),
            plan_query=plan_query.detach(),
            agent_features=agent_features.detach(),
            map_features=map_features.detach(),
            last_final_planning=(
                None
                if last_final_planning is None
                else last_final_planning.detach()
            ),
            history_valid=history_valid,
            ego_status=None if ego_status is None else ego_status.detach(),
        )
        batch_size = base_logits.shape[0]
        base_by_command = base_logits.reshape(
            batch_size, self.num_commands, self.num_modes
        )
        detached_base = base_by_command.detach()
        base_mode = detached_base.argmax(dim=-1)
        relative = self._relative_features(
            soft_output["candidate_features"],
            soft_output["score_metrics"],
            detached_base,
            base_mode,
        )
        takeover_logits = self.takeover_head(relative).squeeze(-1)
        takeover_probability = takeover_logits.sigmoid()
        guard = self._predicted_guard(soft_output["score_metrics"], base_mode)
        mode_index = torch.arange(
            self.num_modes, device=base_logits.device
        ).view(1, 1, self.num_modes)
        alternative = mode_index != base_mode.unsqueeze(-1)
        eligible = (
            alternative
            & guard["all"]
            & (takeover_probability >= self.takeover_probability_threshold)
        )
        masked_takeover = takeover_logits.masked_fill(~eligible, -1e9)
        takeover_mode = masked_takeover.argmax(dim=-1)
        takeover_applied = eligible.any(dim=-1)
        one_hot = F.one_hot(takeover_mode, num_classes=self.num_modes).to(
            base_logits.dtype
        )
        base_value = self._gather_mode(base_by_command, base_mode)
        target_value = base_value.unsqueeze(-1) + self.takeover_logit_epsilon
        boost = F.relu(target_value - base_by_command) * one_hot
        boost = boost * takeover_applied.to(base_logits.dtype).unsqueeze(-1)
        calibrated_by_command = base_by_command + boost
        calibrated_logits = calibrated_by_command.reshape(
            batch_size, 1, self.num_candidates
        )

        output = dict(soft_output)
        output.update(
            {
                "base_logits": base_logits,
                "calibrated_logits": calibrated_logits,
                "score_delta": calibrated_logits - base_logits,
                "ranking_logits": soft_output["calibrated_logits"],
                "ranking_score_delta": soft_output["score_delta"],
                "base_mode": base_mode,
                "takeover_logits": takeover_logits,
                "takeover_probability": takeover_probability,
                "takeover_mode": takeover_mode,
                "takeover_applied": takeover_applied,
                "takeover_eligible": eligible,
                "predicted_guard_all": guard["all"],
                "predicted_guard_l2": guard["l2"],
                "predicted_guard_collision": guard["collision"],
                "predicted_guard_consistency": guard["consistency"],
                "predicted_guard_weighted": guard["weighted"],
                "predicted_weighted_ratio": guard["weighted_ratio"],
                "refined_plan_reg": plan_reg,
            }
        )
        for name, value in output.items():
            if torch.is_tensor(value) and value.is_floating_point():
                if not torch.isfinite(value).all():
                    raise FloatingPointError("CSPC output %s contains NaN/Inf" % name)
        return output

    def _takeover_targets(
        self,
        output: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        metric = targets["metric_target"]
        valid = targets["metric_valid"]
        batch_size = metric.shape[0]
        command = targets["command"]
        batch_index = torch.arange(batch_size, device=metric.device)
        base_by_command = output["base_logits"].detach().reshape(
            batch_size, self.num_commands, self.num_modes
        )
        base_mode = base_by_command[batch_index, command].argmax(dim=-1)
        row = torch.arange(batch_size, device=metric.device)
        baseline = metric[row, base_mode].unsqueeze(1)
        baseline_valid = valid[row, base_mode].unsqueeze(1)
        all_valid = valid[..., :9].all(dim=-1) & baseline_valid[..., :9].all(dim=-1)
        l2_guard = (
            self._safe_ratio(
                metric[..., list(L2_INDICES)],
                baseline[..., list(L2_INDICES)],
            )
            <= 1.0
        ).all(dim=-1)
        collision_guard = (
            metric[..., list(COLLISION_INDICES)]
            <= baseline[..., list(COLLISION_INDICES)] + 1e-8
        ).all(dim=-1)
        consistency_guard = (
            self._safe_ratio(
                metric[..., list(CONSISTENCY_INDICES)],
                baseline[..., list(CONSISTENCY_INDICES)],
            )
            <= 1.01
        ).all(dim=-1)
        weights = self.teacher_metric_weights.to(
            device=metric.device, dtype=metric.dtype
        ).view(1, 1, -1)
        weighted_ratio = (self._safe_ratio(metric, baseline) * weights).sum(dim=-1)
        weighted_guard = weighted_ratio <= 0.98
        alternative = torch.arange(
            self.num_modes, device=metric.device
        ).view(1, self.num_modes) != base_mode.unsqueeze(-1)
        positive = (
            all_valid
            & alternative
            & l2_guard
            & collision_guard
            & consistency_guard
            & weighted_guard
        )
        sample_valid = targets["rank_valid"]
        return {
            "positive": positive,
            "valid": sample_valid.unsqueeze(-1).expand_as(positive),
            "base_mode": base_mode,
            "weighted_ratio": weighted_ratio,
        }

    def loss(
        self,
        output: Dict[str, torch.Tensor],
        data: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        targets = self.build_targets(output, data)
        parent_losses = super().loss(output, data, targets=targets)
        # The parent supplies reusable dense-critic objectives, but this branch
        # is CSPC (the unrestricted CDRS module is disabled). Keep formal logs
        # unambiguous by exposing every reused term under the CSPC namespace.
        losses = {
            name.replace("cdrs_", "cspc_", 1): value
            for name, value in parent_losses.items()
        }
        takeover_target = self._takeover_targets(output, targets)
        batch_size = output["base_logits"].shape[0]
        batch_index = torch.arange(
            batch_size, device=output["base_logits"].device
        )
        command = targets["command"]
        logits = output["takeover_logits"][batch_index, command]
        label = takeover_target["positive"].to(logits.dtype)
        valid = takeover_target["valid"].to(logits.dtype)
        weight = 1.0 + (self.takeover_positive_weight - 1.0) * label
        bce = F.binary_cross_entropy_with_logits(logits, label, reduction="none")
        takeover_loss = (bce * weight * valid).sum() / (
            weight * valid
        ).sum().clamp_min(1.0)
        applied = output["takeover_applied"][batch_index, command]
        eligible = output["takeover_eligible"][batch_index, command]
        losses.update(
            {
                "cspc_loss_takeover": takeover_loss * self.takeover_loss_weight,
                "cspc_stat_target_positive_rate": (
                    (label * valid).sum() / valid.sum().clamp_min(1.0)
                ).detach(),
                "cspc_stat_positive_sample_rate": (
                    takeover_target["positive"].any(dim=-1).to(logits.dtype).mean()
                ).detach(),
                "cspc_stat_predicted_eligible_rate": (
                    eligible.to(logits.dtype).mean()
                ).detach(),
                "cspc_stat_takeover_rate": applied.to(logits.dtype).mean().detach(),
            }
        )
        for name, value in losses.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError("CSPC loss %s contains NaN/Inf" % name)
        return losses


class CSPCPlanningAdapter(CausalSelectiveParetoController):
    """Configuration-facing alias for the selective controller."""
