from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


RCPC_METRIC_NAMES = (
    "l2_4s",
    "l2_5s",
    "l2_6s",
    "avoidable_collision_6s",
    "consistency_4s",
    "consistency_5s",
    "consistency_6s",
    "comfort_6s",
)


def cumulative_to_increments(cumulative: torch.Tensor) -> torch.Tensor:
    if cumulative.shape[-1] != 2:
        raise ValueError("RCPC trajectories must end in xy=2")
    return torch.cat(
        [cumulative[..., :1, :], cumulative[..., 1:, :] - cumulative[..., :-1, :]],
        dim=-2,
    )


def _masked_horizon_mean(
    value: torch.Tensor,
    mask: torch.Tensor,
    horizon: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    value_h = value[..., :horizon]
    mask_h = mask[..., :horizon].to(device=value.device, dtype=value.dtype)
    denom = mask_h.sum(dim=-1).clamp_min(1.0)
    mean = (value_h * mask_h).sum(dim=-1) / denom
    valid = mask_h.sum(dim=-1) > 0
    return mean, valid


def _batch_item(value, index: int, batch_size: int):
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        return value[index]
    if torch.is_tensor(value) and value.dim() > 0 and value.shape[0] == batch_size:
        return value[index]
    return value


def _to_tensor(value, reference: torch.Tensor) -> Optional[torch.Tensor]:
    if value is None:
        return None
    if hasattr(value, "tensor"):
        value = value.tensor
    if torch.is_tensor(value):
        return value.to(device=reference.device, dtype=reference.dtype)
    return torch.as_tensor(value, device=reference.device, dtype=reference.dtype)


def _trajectory_yaw(cumulative: torch.Tensor) -> torch.Tensor:
    increments = cumulative_to_increments(cumulative)
    yaw = torch.atan2(increments[..., 1], increments[..., 0])
    stationary = torch.linalg.norm(increments, dim=-1) < 1e-3
    return torch.where(stationary, yaw.new_full((), math.pi / 2.0), yaw)


def _axis_aligned_half_extents(
    length: torch.Tensor,
    width: torch.Tensor,
    yaw: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    cos_yaw = torch.cos(yaw).abs()
    sin_yaw = torch.sin(yaw).abs()
    half_x = 0.5 * (cos_yaw * length + sin_yaw * width)
    half_y = 0.5 * (sin_yaw * length + cos_yaw * width)
    return half_x, half_y


def avoidable_collision_proxy(
    candidates: torch.Tensor,
    gt_ego: torch.Tensor,
    gt_ego_mask: torch.Tensor,
    gt_boxes,
    gt_agent_fut_trajs,
    gt_agent_fut_masks,
    margin: float = 0.15,
) -> Tuple[torch.Tensor, bool]:
    """Return evaluator-aligned, non-differentiable collision labels.

    The official evaluator ignores collisions already made by the GT ego path.
    This proxy mirrors that rule and uses conservative oriented-box AABBs.  It
    is deliberately used only to construct training targets, never at forward.
    """
    num_modes, horizon, _ = candidates.shape
    boxes = _to_tensor(gt_boxes, candidates)
    agent_fut = _to_tensor(gt_agent_fut_trajs, candidates)
    agent_mask = _to_tensor(gt_agent_fut_masks, candidates)
    if boxes is None or agent_fut is None or agent_mask is None:
        return candidates.new_zeros(num_modes), False
    if boxes.numel() == 0 or agent_fut.numel() == 0:
        return candidates.new_zeros(num_modes), True
    if boxes.dim() != 2 or boxes.shape[-1] < 2:
        return candidates.new_zeros(num_modes), False
    if agent_fut.dim() != 3 or agent_fut.shape[-1] != 2:
        return candidates.new_zeros(num_modes), False

    num_agents = min(boxes.shape[0], agent_fut.shape[0], agent_mask.shape[0])
    boxes = boxes[:num_agents]
    agent_fut = agent_fut[:num_agents, :horizon]
    agent_mask = agent_mask[:num_agents, :horizon].bool()
    if agent_fut.shape[1] < horizon:
        pad = horizon - agent_fut.shape[1]
        agent_fut = F.pad(agent_fut, (0, 0, 0, pad))
        agent_mask = F.pad(agent_mask, (0, pad), value=False)

    agent_centers = boxes[:, None, :2] + agent_fut.cumsum(dim=-2)
    if boxes.shape[-1] >= 7:
        agent_length = boxes[:, 3].abs().clamp_min(0.2)
        agent_width = boxes[:, 4].abs().clamp_min(0.2)
        agent_yaw = boxes[:, 6]
    else:
        agent_length = boxes.new_full((num_agents,), 4.0)
        agent_width = boxes.new_full((num_agents,), 1.8)
        agent_yaw = boxes.new_zeros(num_agents)
    agent_half_x, agent_half_y = _axis_aligned_half_extents(
        agent_length[:, None], agent_width[:, None], agent_yaw[:, None]
    )

    def collision_by_time(ego_paths: torch.Tensor) -> torch.Tensor:
        ego_yaw = _trajectory_yaw(ego_paths)
        ego_length = ego_paths.new_full(ego_yaw.shape, 4.084)
        ego_width = ego_paths.new_full(ego_yaw.shape, 1.85)
        ego_half_x, ego_half_y = _axis_aligned_half_extents(
            ego_length, ego_width, ego_yaw
        )
        delta = ego_paths[:, None] - agent_centers[None]
        overlap_x = delta[..., 0].abs() <= (
            ego_half_x[:, None] + agent_half_x[None] + float(margin)
        )
        overlap_y = delta[..., 1].abs() <= (
            ego_half_y[:, None] + agent_half_y[None] + float(margin)
        )
        overlap = overlap_x & overlap_y & agent_mask[None]
        return overlap.any(dim=1)

    candidate_collision = collision_by_time(candidates)
    gt_collision = collision_by_time(gt_ego.unsqueeze(0))[0]
    valid_time = gt_ego_mask[:horizon].bool()
    avoidable = candidate_collision & (~gt_collision.unsqueeze(0))
    denom = valid_time.to(candidates.dtype).sum().clamp_min(1.0)
    cost = (
        avoidable.to(candidates.dtype)
        * valid_time.to(candidates.dtype).unsqueeze(0)
    ).sum(dim=-1) / denom
    return cost, True


class CausalInnovationParetoSelector(nn.Module):
    """C4-D set-wise selector conditioned on strictly causal innovation.

    The selector treats the frozen C4-C decision as an incumbent and evaluates
    all six command-compatible candidates jointly.  Its forward path consumes
    model tensors, analytic trajectory geometry and retrospective errors that
    are already observable at the current frame.  Future GT is intentionally
    absent; it is used only by the parent ``loss`` method.

    A zero-initialized distribution head makes the lower confidence bounds
    negative at initialization, so enabling C4-D is an exact behavioral no-op
    until learned evidence supports a Pareto-safe replacement.
    """

    def __init__(
        self,
        token_dims: int,
        num_commands: int = 3,
        num_modes: int = 6,
        horizon_steps: int = 12,
        num_heads: int = 6,
        dropout: float = 0.0,
        lcb_margins: Sequence[float] = (0.0, 0.0, 0.0, -0.02),
        calibration_offsets: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        teacher_margins: Sequence[float] = (0.005, 0.005, 0.005, 0.0),
        analytic_tolerances: Sequence[float] = (0.0, 0.0, 0.0, 0.02),
        utility_weights: Sequence[float] = (0.15, 0.25, 0.50, 0.10),
        utility_residual_scale: float = 0.05,
        max_mean_displacement: float = 4.0,
        max_endpoint_displacement: float = 8.0,
        score_gap: float = 1.0,
        require_history: bool = True,
        require_calibration: bool = True,
        enable_direct_takeover_head: bool = False,
        takeover_threshold: float = 1.0,
        takeover_utility_scale: float = 0.10,
        enable_margin_calibrator_head: bool = False,
        margin_threshold: float = 1.0,
        margin_evidence_scale: float = 1.0,
        margin_evidence_temperature: float = 0.05,
        margin_utility_scale: float = 0.10,
    ) -> None:
        super().__init__()
        self.token_dims = int(token_dims)
        self.num_commands = int(num_commands)
        self.num_modes = int(num_modes)
        self.horizon_steps = int(horizon_steps)
        self.utility_residual_scale = float(utility_residual_scale)
        self.max_mean_displacement = float(max_mean_displacement)
        self.max_endpoint_displacement = float(max_endpoint_displacement)
        self.score_gap = float(score_gap)
        self.require_history = bool(require_history)
        self.require_calibration = bool(require_calibration)
        self.enable_direct_takeover_head = bool(enable_direct_takeover_head)
        self.takeover_threshold = float(takeover_threshold)
        self.takeover_utility_scale = float(takeover_utility_scale)
        self.enable_margin_calibrator_head = bool(
            enable_margin_calibrator_head
        )
        self.margin_threshold = float(margin_threshold)
        self.margin_evidence_scale = float(margin_evidence_scale)
        self.margin_evidence_temperature = float(
            margin_evidence_temperature
        )
        self.margin_utility_scale = float(margin_utility_scale)
        if self.token_dims % int(num_heads) != 0:
            raise ValueError("C4-D token_dims must be divisible by num_heads")
        if self.num_commands != 3 or self.num_modes != 6:
            raise ValueError("C4-D expects MomAD's 3 commands x 6 modes")
        if self.horizon_steps != 12:
            raise ValueError("C4-D expects the audited 12-step planning horizon")
        if any(
            len(values) != 4
            for values in (
                lcb_margins,
                calibration_offsets,
                teacher_margins,
                analytic_tolerances,
                utility_weights,
            )
        ):
            raise ValueError("C4-D metric vectors must each contain four values")
        if self.max_mean_displacement <= 0.0:
            raise ValueError("C4-D mean displacement cap must be positive")
        if self.max_endpoint_displacement <= 0.0:
            raise ValueError("C4-D endpoint displacement cap must be positive")
        if self.score_gap <= 0.0:
            raise ValueError("C4-D score gap must be positive")
        if not 0.0 <= self.takeover_threshold <= 1.0:
            raise ValueError("C4-E takeover threshold must lie in [0, 1]")
        if self.takeover_utility_scale < 0.0:
            raise ValueError("C4-E takeover utility scale must be non-negative")
        if self.enable_margin_calibrator_head and not self.enable_direct_takeover_head:
            raise ValueError(
                "C4-F margin calibrator requires the frozen C4-E direct head"
            )
        if not 0.0 <= self.margin_threshold <= 1.0:
            raise ValueError("C4-F margin threshold must lie in [0, 1]")
        if self.margin_evidence_scale < 0.0:
            raise ValueError("C4-F margin evidence scale must be non-negative")
        if self.margin_evidence_temperature <= 0.0:
            raise ValueError("C4-F margin evidence temperature must be positive")
        if self.margin_utility_scale < 0.0:
            raise ValueError("C4-F margin utility scale must be non-negative")

        self.register_buffer(
            "lcb_margins",
            torch.tensor(lcb_margins, dtype=torch.float32),
            persistent=False,
        )
        # Keep post-training conformal offsets out of checkpoints. MMCV's
        # checkpoint helper serializes even PyTorch non-persistent buffers,
        # which would otherwise overwrite a calibration config at load time.
        self.calibration_offsets = tuple(
            float(value) for value in calibration_offsets
        )
        self.register_buffer(
            "teacher_margins",
            torch.tensor(teacher_margins, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "analytic_tolerances",
            torch.tensor(analytic_tolerances, dtype=torch.float32),
            persistent=False,
        )
        utility = torch.tensor(utility_weights, dtype=torch.float32)
        if (utility < 0).any() or utility.sum() <= 0:
            raise ValueError("C4-D utility weights must be non-negative and non-zero")
        self.register_buffer(
            "utility_weights", utility / utility.sum(), persistent=False
        )
        self.register_buffer(
            "innovation_scales",
            torch.tensor(
                [10.0, 10.0, math.pi, 10.0, 20.0, 2.0],
                dtype=torch.float32,
            ),
            persistent=False,
        )

        # Four candidate/scene tokens plus 19 causal scalar comparisons:
        # two score contrasts, eight critic-metric contrasts, four analytic
        # contrasts, three displacement summaries and two observability flags.
        pair_dims = 4 * self.token_dims + 19
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(pair_dims),
            nn.Linear(pair_dims, self.token_dims * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.token_dims * 2, self.token_dims),
        )
        # Six retrospective innovation values, four availability bits and the
        # audited history-continuity bit form the explicit causal innovation.
        self.innovation_encoder = nn.Sequential(
            nn.Linear(11, self.token_dims),
            nn.GELU(),
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims),
        )
        self.innovation_gate = nn.Sequential(
            nn.LayerNorm(self.token_dims * 2),
            nn.Linear(self.token_dims * 2, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, 1),
        )
        self.set_attention = nn.MultiheadAttention(
            self.token_dims,
            int(num_heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.set_norm = nn.LayerNorm(self.token_dims)
        self.set_ffn = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.token_dims * 2, self.token_dims),
        )
        self.output_norm = nn.LayerNorm(self.token_dims)
        self.improvement_head = nn.Sequential(
            nn.Linear(self.token_dims, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, 8),
        )
        self.utility_head = nn.Sequential(
            nn.Linear(self.token_dims, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, 1),
        )
        self.takeover_head = None
        if self.enable_direct_takeover_head:
            # Frozen set token + 4 medians + 4 lower bounds + 4 causal
            # analytic improvements + scalar utility + three displacement
            # summaries + retrospective innovation gate.
            takeover_dims = self.token_dims + 17
            self.takeover_head = nn.Sequential(
                nn.LayerNorm(takeover_dims),
                nn.Linear(takeover_dims, self.token_dims),
                nn.GELU(),
                nn.Linear(self.token_dims, 1),
            )
        self.margin_calibrator_head = None
        if self.enable_margin_calibrator_head:
            # C4-F consumes the complete frozen C4-E takeover representation
            # plus its scalar logit.  Four outputs estimate dense paired
            # margins (L2@4/5/6s and collision); the fifth is a joint-safety
            # residual.  No future target enters this forward representation.
            margin_dims = self.token_dims + 18
            self.margin_calibrator_head = nn.Sequential(
                nn.LayerNorm(margin_dims),
                nn.Linear(margin_dims, self.token_dims),
                nn.GELU(),
                nn.Linear(self.token_dims, 5),
            )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # Conservative no-op start: median=0 and spread=softplus(0), hence no
        # alternative can satisfy a non-negative simultaneous LCB gate.
        nn.init.zeros_(self.improvement_head[-1].weight)
        nn.init.zeros_(self.improvement_head[-1].bias)
        nn.init.zeros_(self.utility_head[-1].weight)
        nn.init.zeros_(self.utility_head[-1].bias)
        nn.init.zeros_(self.innovation_gate[-1].weight)
        nn.init.constant_(self.innovation_gate[-1].bias, -1.0)
        if self.takeover_head is not None:
            # Probability starts at 0.5 and the locked threshold starts at
            # 1.0, making C4-E an exact incumbent fallback before calibration.
            nn.init.zeros_(self.takeover_head[-1].weight)
            nn.init.zeros_(self.takeover_head[-1].bias)
        if self.margin_calibrator_head is not None:
            # Threshold one is an exact fallback, while zero output weights
            # make initialization independent of random hidden features.
            nn.init.zeros_(self.margin_calibrator_head[-1].weight)
            nn.init.zeros_(self.margin_calibrator_head[-1].bias)

    @staticmethod
    def _gather_mode(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
        batch_size, num_commands = index.shape
        extra_shape = value.shape[3:]
        gather_index = index[..., None]
        for _ in extra_shape:
            gather_index = gather_index.unsqueeze(-1)
        gather_index = gather_index.expand(
            batch_size, num_commands, 1, *extra_shape
        )
        return torch.gather(value, 2, gather_index).squeeze(2)

    def _innovation_token(
        self,
        reference: torch.Tensor,
        calibration_features: Optional[torch.Tensor],
        calibration_availability: Optional[torch.Tensor],
        history_valid: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = reference.shape[0]
        if calibration_features is None:
            values = reference.new_zeros(batch_size, 6)
        else:
            values = calibration_features.detach().to(
                device=reference.device, dtype=reference.dtype
            ).reshape(batch_size, 6)
        scales = self.innovation_scales.to(
            device=reference.device, dtype=reference.dtype
        ).view(1, 6)
        values = torch.clamp(values / scales, -5.0, 5.0)
        if calibration_availability is None:
            availability = reference.new_zeros(batch_size, 4)
        else:
            availability = calibration_availability.detach().to(
                device=reference.device, dtype=reference.dtype
            ).reshape(batch_size, 4)
            availability = (availability > 0).to(reference.dtype)
        if history_valid is None:
            history = reference.new_zeros(batch_size, 1)
        else:
            history = history_valid.detach().to(device=reference.device).reshape(
                batch_size, 1
            )
            history = history.to(reference.dtype)
        raw = torch.cat([values, availability, history], dim=-1)
        return self.innovation_encoder(raw), history.bool().squeeze(-1), (
            availability.bool().any(dim=-1)
        )

    def forward(
        self,
        candidate_tokens: torch.Tensor,
        command_context: torch.Tensor,
        calibration_context: torch.Tensor,
        trajectories: torch.Tensor,
        incumbent_logits: torch.Tensor,
        effective_metrics: torch.Tensor,
        analytic_metrics: torch.Tensor,
        calibration_features: Optional[torch.Tensor] = None,
        calibration_availability: Optional[torch.Tensor] = None,
        history_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = candidate_tokens.shape[0]
        expected_tokens = (
            batch_size,
            self.num_commands,
            self.num_modes,
            self.token_dims,
        )
        if tuple(candidate_tokens.shape) != expected_tokens:
            raise ValueError(
                "C4-D candidate tokens must be %s, got %s"
                % (expected_tokens, tuple(candidate_tokens.shape))
            )
        if tuple(incumbent_logits.shape) != (
            batch_size,
            self.num_commands,
            self.num_modes,
        ):
            raise ValueError("C4-D incumbent logits have an invalid shape")

        incumbent_winner = incumbent_logits.detach().argmax(dim=-1)
        incumbent_token = self._gather_mode(
            candidate_tokens, incumbent_winner
        )
        incumbent_trajectory = self._gather_mode(
            trajectories, incumbent_winner
        )
        incumbent_effective = self._gather_mode(
            effective_metrics, incumbent_winner
        )
        incumbent_analytic = self._gather_mode(
            analytic_metrics, incumbent_winner
        )
        incumbent_score = torch.gather(
            incumbent_logits, 2, incumbent_winner[..., None]
        ).squeeze(-1)

        displacement = torch.linalg.norm(
            trajectories - incumbent_trajectory[:, :, None], dim=-1
        )
        effective_improvement = (
            incumbent_effective[:, :, None] - effective_metrics
        )
        analytic_improvement = (
            incumbent_analytic[:, :, None] - analytic_metrics
        )
        centered_score = incumbent_logits - incumbent_logits.mean(
            dim=-1, keepdim=True
        )
        score_vs_incumbent = incumbent_logits - incumbent_score[..., None]

        if history_valid is None:
            history_flag = candidate_tokens.new_zeros(batch_size)
        else:
            history_flag = (
                history_valid.detach()
                .to(candidate_tokens.device)
                .bool()
                .reshape(batch_size)
                .to(candidate_tokens.dtype)
            )
        if calibration_availability is None:
            calibration_flag = candidate_tokens.new_zeros(batch_size)
        else:
            calibration_flag = (
                calibration_availability.detach()
                .to(candidate_tokens.device)
                .reshape(batch_size, 4)
                .bool()
                .any(dim=-1)
                .to(candidate_tokens.dtype)
            )
        observability = torch.stack(
            [history_flag, calibration_flag], dim=-1
        )[:, None, None].expand(-1, self.num_commands, self.num_modes, -1)
        scalar = torch.cat(
            [
                centered_score[..., None],
                score_vs_incumbent[..., None],
                effective_improvement,
                analytic_improvement,
                displacement.mean(dim=-1, keepdim=True),
                displacement[..., -1:].contiguous(),
                displacement.max(dim=-1, keepdim=True).values,
                observability,
            ],
            dim=-1,
        )
        command_token = command_context[:, :, None].expand(
            -1, -1, self.num_modes, -1
        )
        calibration_token = calibration_context[:, None, None].expand(
            -1, self.num_commands, self.num_modes, -1
        )
        pair_input = torch.cat(
            [
                candidate_tokens,
                candidate_tokens - incumbent_token[:, :, None],
                command_token,
                calibration_token,
                scalar,
            ],
            dim=-1,
        )
        pair_token = self.pair_encoder(pair_input)
        innovation_token, history_available, calibration_available = (
            self._innovation_token(
                pair_token,
                calibration_features,
                calibration_availability,
                history_valid,
            )
        )
        innovation_expanded = innovation_token[:, None, None].expand(
            -1, self.num_commands, self.num_modes, -1
        )
        innovation_gate = torch.sigmoid(
            self.innovation_gate(
                torch.cat([pair_token, innovation_expanded], dim=-1)
            )
        )
        fused = pair_token + innovation_gate * innovation_expanded
        set_input = fused.reshape(
            batch_size * self.num_commands, self.num_modes, self.token_dims
        )
        set_update, _ = self.set_attention(
            set_input, set_input, set_input, need_weights=False
        )
        set_token = self.set_norm(set_input + set_update)
        set_token = set_token + self.set_ffn(set_token)
        set_token = self.output_norm(set_token).reshape(
            batch_size,
            self.num_commands,
            self.num_modes,
            self.token_dims,
        )

        raw = self.improvement_head(set_token)
        metric_median = raw[..., :4]
        metric_spread = F.softplus(raw[..., 4:])
        metric_lcb = metric_median - metric_spread
        utility_residual = self.utility_head(set_token).squeeze(-1)
        margin = self.lcb_margins.to(
            device=metric_lcb.device, dtype=metric_lcb.dtype
        ) + metric_lcb.new_tensor(self.calibration_offsets)
        margin = margin.view(1, 1, 1, 4)
        predicted_safe = (metric_lcb >= margin).all(dim=-1)
        tolerance = self.analytic_tolerances.to(
            device=analytic_improvement.device,
            dtype=analytic_improvement.dtype,
        ).view(1, 1, 1, 4)
        structural_safe = (
            analytic_improvement >= -tolerance
        ).all(dim=-1)
        structural_safe = structural_safe & (
            displacement.mean(dim=-1) <= self.max_mean_displacement
        )
        structural_safe = structural_safe & (
            displacement[..., -1] <= self.max_endpoint_displacement
        )
        observability_safe = torch.ones_like(structural_safe)
        if self.require_history:
            observability_safe = observability_safe & history_available[
                :, None, None
            ]
        if self.require_calibration:
            observability_safe = observability_safe & calibration_available[
                :, None, None
            ]
        mode_index = torch.arange(
            self.num_modes, device=incumbent_winner.device
        ).view(1, 1, self.num_modes)
        changed = mode_index != incumbent_winner[..., None]
        eligible = (
            predicted_safe & structural_safe & observability_safe & changed
        )
        utility_weight = self.utility_weights.to(
            device=metric_lcb.device, dtype=metric_lcb.dtype
        ).view(1, 1, 1, 4)
        candidate_utility = (metric_lcb * utility_weight).sum(dim=-1)
        candidate_utility = candidate_utility + self.utility_residual_scale * torch.tanh(
            utility_residual
        )
        takeover_logit = None
        takeover_probability = None
        takeover_proposal_score = None
        direct_eligible = None
        direct_proposed_winner = None
        margin_prediction = None
        margin_joint_logit = None
        margin_evidence = None
        margin_safety_score = None
        margin_probability = None
        if self.takeover_head is not None:
            takeover_input = torch.cat(
                [
                    set_token,
                    metric_median,
                    metric_lcb,
                    analytic_improvement,
                    candidate_utility[..., None],
                    displacement.mean(dim=-1, keepdim=True),
                    displacement[..., -1:].contiguous(),
                    displacement.max(dim=-1, keepdim=True).values,
                    innovation_gate,
                ],
                dim=-1,
            )
            takeover_logit = self.takeover_head(takeover_input).squeeze(-1)
            takeover_probability = torch.sigmoid(takeover_logit)
            direct_eligible = structural_safe & observability_safe & changed
            takeover_proposal_score = (
                takeover_logit
                + self.takeover_utility_scale * torch.tanh(candidate_utility)
            )
            masked_takeover_score = takeover_proposal_score.masked_fill(
                ~direct_eligible,
                torch.finfo(takeover_proposal_score.dtype).min,
            )
            direct_proposed_winner = masked_takeover_score.argmax(dim=-1)
            proposed_winner = direct_proposed_winner
            proposed_probability = torch.gather(
                takeover_probability, 2, proposed_winner[..., None]
            ).squeeze(-1)
            # Threshold one is the protocol's exact incumbent fallback.  Make
            # that identity explicit instead of relying on sigmoid staying
            # strictly below one after fp16/bf16 rounding.
            if self.takeover_threshold >= 1.0:
                selector_applied = torch.zeros_like(
                    proposed_probability, dtype=torch.bool
                )
            else:
                selector_applied = direct_eligible.any(dim=-1) & (
                    proposed_probability >= self.takeover_threshold
                )
            selector_winner = torch.where(
                selector_applied, proposed_winner, incumbent_winner
            )

            if self.margin_calibrator_head is not None:
                # The C4-D set representation and C4-E direct logit are a
                # frozen feature bank for C4-F.  Detaching here makes that
                # isolation true even outside the prefix-freezing trainer.
                margin_input = torch.cat(
                    [
                        takeover_input.detach(),
                        takeover_logit.detach()[..., None],
                    ],
                    dim=-1,
                )
                margin_raw = self.margin_calibrator_head(margin_input)
                margin_prediction = margin_raw[..., :4]
                margin_joint_logit = margin_raw[..., 4]
                teacher_margin = self.teacher_margins.to(
                    device=margin_prediction.device,
                    dtype=margin_prediction.dtype,
                ).view(1, 1, 1, 4)
                normalized_margin = (
                    margin_prediction - teacher_margin
                ) / self.margin_evidence_temperature
                # A candidate has only as much evidence as its weakest dense
                # safety dimension.  tanh bounds any single extrapolation.
                margin_evidence = torch.tanh(normalized_margin).min(
                    dim=-1
                ).values
                margin_safety_score = (
                    margin_joint_logit
                    + self.margin_evidence_scale * margin_evidence
                    + self.margin_utility_scale
                    * torch.tanh(candidate_utility.detach())
                )
                margin_probability = torch.sigmoid(margin_safety_score)
                masked_margin_score = margin_safety_score.masked_fill(
                    ~direct_eligible,
                    torch.finfo(margin_safety_score.dtype).min,
                )
                proposed_winner = masked_margin_score.argmax(dim=-1)
                proposed_probability = torch.gather(
                    margin_probability, 2, proposed_winner[..., None]
                ).squeeze(-1)
                if self.margin_threshold >= 1.0:
                    selector_applied = torch.zeros_like(
                        proposed_probability, dtype=torch.bool
                    )
                else:
                    selector_applied = direct_eligible.any(dim=-1) & (
                        proposed_probability >= self.margin_threshold
                    )
                selector_winner = torch.where(
                    selector_applied, proposed_winner, incumbent_winner
                )
        else:
            masked_utility = candidate_utility.masked_fill(
                ~eligible, torch.finfo(candidate_utility.dtype).min
            )
            proposed_winner = masked_utility.argmax(dim=-1)
            selector_applied = eligible.any(dim=-1)
            selector_winner = torch.where(
                selector_applied, proposed_winner, incumbent_winner
            )

        winner_score = torch.gather(
            incumbent_logits, 2, selector_winner[..., None]
        ).squeeze(-1)
        desired_score = incumbent_logits.max(dim=-1).values + self.score_gap
        score_boost = (desired_score - winner_score).clamp_min(0.0)
        proposed_logits = incumbent_logits.scatter_add(
            2, selector_winner[..., None], score_boost[..., None]
        )
        calibrated_logits = torch.where(
            selector_applied[..., None], proposed_logits, incumbent_logits
        )
        output = {
            "calibrated_logits": calibrated_logits,
            "incumbent_winner": incumbent_winner,
            "selector_winner": selector_winner,
            "selector_applied": selector_applied,
            "metric_median": metric_median,
            "metric_spread": metric_spread,
            "metric_lcb": metric_lcb,
            "predicted_safe": predicted_safe,
            "structural_safe": structural_safe,
            "observability_safe": observability_safe,
            "eligible": eligible,
            "analytic_improvement": analytic_improvement,
            "candidate_displacement": displacement,
            "candidate_utility": candidate_utility,
            "utility_residual": utility_residual,
            "innovation_gate": innovation_gate.squeeze(-1),
        }
        if takeover_logit is not None:
            output.update(
                {
                    "takeover_logit": takeover_logit,
                    "takeover_probability": takeover_probability,
                    "takeover_proposal_score": takeover_proposal_score,
                    "direct_eligible": direct_eligible,
                    "direct_proposed_winner": direct_proposed_winner,
                    "proposed_winner": proposed_winner,
                }
            )
        if margin_prediction is not None:
            output.update(
                {
                    "margin_prediction": margin_prediction,
                    "margin_joint_logit": margin_joint_logit,
                    "margin_evidence": margin_evidence,
                    "margin_safety_score": margin_safety_score,
                    "margin_probability": margin_probability,
                }
            )
        return output


class RetrospectivelyCalibratedParetoCritic(nn.Module):
    """Causal candidate critic with adaptive multi-objective ranking.

    Forward consumes only model predictions, current scene features, observed
    ego status and cached model history.  GT future tensors are accepted only
    by :meth:`loss`, where they form dense teacher targets.
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
        detach_inputs: bool = False,
        enable_refiner: bool = False,
        top_m: int = 3,
        max_refine_offset: float = 1.5,
        max_score_delta: float = 8.0,
        score_blend: float = 1.0,
        enable_trust_gate: bool = False,
        trust_gate_threshold: float = 0.70,
        trust_gate_teacher_margin: float = 0.01,
        trust_gate_loss_weight: float = 1.00,
        trust_gate_positive_weight: float = 2.00,
        enable_risk_controller: bool = False,
        risk_quantile: float = 0.10,
        risk_lcb_margins: Sequence[float] = (0.0, 0.0, 0.0, -0.02),
        risk_lcb_calibration_offsets: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        risk_teacher_margins: Sequence[float] = (0.01, 0.01, 0.01, 0.0),
        risk_analytic_tolerance: float = 0.0,
        risk_max_mean_displacement: float = 1.0,
        risk_max_endpoint_displacement: float = 2.0,
        risk_median_loss_weight: float = 0.50,
        risk_quantile_loss_weight: float = 1.00,
        risk_decision_loss_weight: float = 0.50,
        risk_false_takeover_weight: float = 8.00,
        risk_decision_temperature: float = 0.05,
        enable_causal_pareto_selector: bool = False,
        selector_quantile: float = 0.10,
        selector_lcb_margins: Sequence[float] = (0.0, 0.0, 0.0, -0.02),
        selector_calibration_offsets: Sequence[float] = (0.0, 0.0, 0.0, 0.0),
        selector_teacher_margins: Sequence[float] = (0.005, 0.005, 0.005, 0.0),
        selector_analytic_tolerances: Sequence[float] = (0.0, 0.0, 0.0, 0.02),
        selector_utility_weights: Sequence[float] = (0.15, 0.25, 0.50, 0.10),
        selector_utility_residual_scale: float = 0.05,
        selector_max_mean_displacement: float = 4.0,
        selector_max_endpoint_displacement: float = 8.0,
        selector_score_gap: float = 1.0,
        selector_require_history: bool = True,
        selector_require_calibration: bool = True,
        selector_enable_direct_takeover_head: bool = False,
        selector_takeover_threshold: float = 1.0,
        selector_takeover_utility_scale: float = 0.10,
        selector_enable_margin_calibrator_head: bool = False,
        selector_margin_threshold: float = 1.0,
        selector_margin_evidence_scale: float = 1.0,
        selector_margin_evidence_temperature: float = 0.05,
        selector_margin_utility_scale: float = 0.10,
        selector_median_loss_weight: float = 0.50,
        selector_quantile_loss_weight: float = 1.00,
        selector_decision_loss_weight: float = 0.50,
        selector_listwise_loss_weight: float = 0.50,
        selector_false_takeover_weight: float = 8.00,
        selector_positive_weight: float = 2.00,
        selector_decision_temperature: float = 0.05,
        selector_listwise_temperature: float = 0.10,
        selector_direct_safety_loss_weight: float = 1.00,
        selector_direct_rank_loss_weight: float = 0.25,
        selector_direct_negative_weight: float = 2.00,
        selector_direct_rank_margin: float = 1.00,
        selector_margin_regression_loss_weight: float = 1.00,
        selector_margin_joint_loss_weight: float = 1.00,
        selector_margin_positive_weight: float = 8.00,
        selector_margin_rank_loss_weight: float = 0.50,
        selector_margin_abstention_loss_weight: float = 0.25,
        selector_margin_rank_margin: float = 1.00,
        selector_margin_abstention_target: float = -1.00,
        score_scale: float = 4.0,
        residual_score_scale: float = 0.25,
        listwise_temperature: float = 0.35,
        pairwise_margin: float = 0.03,
        metric_loss_weight: float = 0.50,
        listwise_loss_weight: float = 1.00,
        pairwise_loss_weight: float = 0.25,
        pareto_prior_loss_weight: float = 0.01,
        refine_loss_weight: float = 0.50,
        residual_loss_weight: float = 0.02,
        metric_scales: Sequence[float] = (2.0, 3.0, 4.0, 1.0, 2.0, 3.0, 4.0, 4.0),
        teacher_metric_weights: Sequence[float] = (
            0.10,
            0.15,
            0.30,
            0.25,
            0.02,
            0.03,
            0.05,
            0.10,
        ),
        pareto_prior: Sequence[float] = (
            0.10,
            0.15,
            0.30,
            0.25,
            0.02,
            0.03,
            0.05,
            0.10,
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
        self.detach_inputs = bool(detach_inputs)
        self.enable_refiner = bool(enable_refiner)
        self.top_m = int(top_m)
        self.max_refine_offset = float(max_refine_offset)
        self.max_score_delta = float(max_score_delta)
        self.score_blend = float(score_blend)
        self.enable_trust_gate = bool(enable_trust_gate)
        self.trust_gate_threshold = float(trust_gate_threshold)
        self.trust_gate_teacher_margin = float(trust_gate_teacher_margin)
        self.trust_gate_loss_weight = float(trust_gate_loss_weight)
        self.trust_gate_positive_weight = float(trust_gate_positive_weight)
        self.enable_risk_controller = bool(enable_risk_controller)
        self.risk_quantile = float(risk_quantile)
        self.risk_lcb_calibration_offsets = tuple(
            float(value) for value in risk_lcb_calibration_offsets
        )
        self.risk_analytic_tolerance = float(risk_analytic_tolerance)
        self.risk_max_mean_displacement = float(risk_max_mean_displacement)
        self.risk_max_endpoint_displacement = float(
            risk_max_endpoint_displacement
        )
        self.risk_median_loss_weight = float(risk_median_loss_weight)
        self.risk_quantile_loss_weight = float(risk_quantile_loss_weight)
        self.risk_decision_loss_weight = float(risk_decision_loss_weight)
        self.risk_false_takeover_weight = float(risk_false_takeover_weight)
        self.risk_decision_temperature = float(risk_decision_temperature)
        self.enable_causal_pareto_selector = bool(
            enable_causal_pareto_selector
        )
        self.selector_quantile = float(selector_quantile)
        self.selector_median_loss_weight = float(selector_median_loss_weight)
        self.selector_quantile_loss_weight = float(selector_quantile_loss_weight)
        self.selector_decision_loss_weight = float(selector_decision_loss_weight)
        self.selector_listwise_loss_weight = float(selector_listwise_loss_weight)
        self.selector_false_takeover_weight = float(
            selector_false_takeover_weight
        )
        self.selector_positive_weight = float(selector_positive_weight)
        self.selector_decision_temperature = float(
            selector_decision_temperature
        )
        self.selector_listwise_temperature = float(
            selector_listwise_temperature
        )
        self.selector_direct_safety_loss_weight = float(
            selector_direct_safety_loss_weight
        )
        self.selector_direct_rank_loss_weight = float(
            selector_direct_rank_loss_weight
        )
        self.selector_direct_negative_weight = float(
            selector_direct_negative_weight
        )
        self.selector_direct_rank_margin = float(
            selector_direct_rank_margin
        )
        self.selector_margin_regression_loss_weight = float(
            selector_margin_regression_loss_weight
        )
        self.selector_margin_joint_loss_weight = float(
            selector_margin_joint_loss_weight
        )
        self.selector_margin_positive_weight = float(
            selector_margin_positive_weight
        )
        self.selector_margin_rank_loss_weight = float(
            selector_margin_rank_loss_weight
        )
        self.selector_margin_abstention_loss_weight = float(
            selector_margin_abstention_loss_weight
        )
        self.selector_margin_rank_margin = float(
            selector_margin_rank_margin
        )
        self.selector_margin_abstention_target = float(
            selector_margin_abstention_target
        )
        self.score_scale = float(score_scale)
        self.residual_score_scale = float(residual_score_scale)
        self.listwise_temperature = float(listwise_temperature)
        self.pairwise_margin = float(pairwise_margin)
        self.metric_loss_weight = float(metric_loss_weight)
        self.listwise_loss_weight = float(listwise_loss_weight)
        self.pairwise_loss_weight = float(pairwise_loss_weight)
        self.pareto_prior_loss_weight = float(pareto_prior_loss_weight)
        self.refine_loss_weight = float(refine_loss_weight)
        self.residual_loss_weight = float(residual_loss_weight)

        if self.horizon_steps != 12:
            raise ValueError("RCPC is defined for the audited MomAD 12-step horizon")
        if self.num_commands != 3 or self.num_modes != 6:
            raise ValueError("RCPC expects MomAD's 3 commands x 6 modes")
        if self.token_dims % int(num_heads) != 0:
            raise ValueError("token_dims must be divisible by num_heads")
        if self.top_m < 1 or self.top_m > self.num_modes:
            raise ValueError("top_m must lie in [1, num_modes]")
        if not 0.0 <= self.score_blend <= 1.0:
            raise ValueError("score_blend must lie in [0, 1]")
        if not 0.0 < self.trust_gate_threshold < 1.0:
            raise ValueError("trust_gate_threshold must lie in (0, 1)")
        if self.trust_gate_teacher_margin < 0.0:
            raise ValueError("trust_gate_teacher_margin must be non-negative")
        if self.trust_gate_loss_weight < 0.0:
            raise ValueError("trust_gate_loss_weight must be non-negative")
        if self.trust_gate_positive_weight < 1.0:
            raise ValueError("trust_gate_positive_weight must be at least one")
        if self.enable_trust_gate and self.enable_risk_controller:
            raise ValueError(
                "trust gate and risk controller are mutually exclusive arbiters"
            )
        if self.enable_causal_pareto_selector and not self.enable_risk_controller:
            raise ValueError(
                "C4-D selector requires the frozen C4-C risk-controller incumbent"
            )
        if not 0.0 < self.risk_quantile < 0.5:
            raise ValueError("risk_quantile must lie in (0, 0.5)")
        if (
            len(risk_lcb_margins) != 4
            or len(self.risk_lcb_calibration_offsets) != 4
            or len(risk_teacher_margins) != 4
        ):
            raise ValueError("risk metric margins must have four entries")
        if self.risk_analytic_tolerance < 0.0:
            raise ValueError("risk_analytic_tolerance must be non-negative")
        if self.risk_max_mean_displacement <= 0.0:
            raise ValueError("risk_max_mean_displacement must be positive")
        if self.risk_max_endpoint_displacement <= 0.0:
            raise ValueError("risk_max_endpoint_displacement must be positive")
        if min(
            self.risk_median_loss_weight,
            self.risk_quantile_loss_weight,
            self.risk_decision_loss_weight,
        ) < 0.0:
            raise ValueError("risk loss weights must be non-negative")
        if self.risk_false_takeover_weight < 1.0:
            raise ValueError("risk_false_takeover_weight must be at least one")
        if self.risk_decision_temperature <= 0.0:
            raise ValueError("risk_decision_temperature must be positive")
        if not 0.0 < self.selector_quantile < 0.5:
            raise ValueError("selector_quantile must lie in (0, 0.5)")
        if min(
            self.selector_median_loss_weight,
            self.selector_quantile_loss_weight,
            self.selector_decision_loss_weight,
            self.selector_listwise_loss_weight,
        ) < 0.0:
            raise ValueError("C4-D selector loss weights must be non-negative")
        if self.selector_false_takeover_weight < 1.0:
            raise ValueError("selector_false_takeover_weight must be at least one")
        if self.selector_positive_weight < 1.0:
            raise ValueError("selector_positive_weight must be at least one")
        if self.selector_decision_temperature <= 0.0:
            raise ValueError("selector_decision_temperature must be positive")
        if self.selector_listwise_temperature <= 0.0:
            raise ValueError("selector_listwise_temperature must be positive")
        if min(
            self.selector_direct_safety_loss_weight,
            self.selector_direct_rank_loss_weight,
        ) < 0.0:
            raise ValueError("C4-E direct safety loss weights must be non-negative")
        if self.selector_direct_negative_weight < 1.0:
            raise ValueError("C4-E direct negative weight must be at least one")
        if self.selector_direct_rank_margin <= 0.0:
            raise ValueError("C4-E direct rank margin must be positive")
        if min(
            self.selector_margin_regression_loss_weight,
            self.selector_margin_joint_loss_weight,
            self.selector_margin_rank_loss_weight,
            self.selector_margin_abstention_loss_weight,
        ) < 0.0:
            raise ValueError("C4-F margin loss weights must be non-negative")
        if self.selector_margin_positive_weight < 1.0:
            raise ValueError("C4-F positive weight must be at least one")
        if self.selector_margin_rank_margin <= 0.0:
            raise ValueError("C4-F rank margin must be positive")
        if not math.isfinite(self.selector_margin_abstention_target):
            raise ValueError("C4-F abstention target must be finite")
        if len(metric_scales) != len(RCPC_METRIC_NAMES):
            raise ValueError("metric_scales must have eight entries")
        if len(teacher_metric_weights) != len(RCPC_METRIC_NAMES):
            raise ValueError("teacher_metric_weights must have eight entries")
        if len(pareto_prior) != len(RCPC_METRIC_NAMES):
            raise ValueError("pareto_prior must have eight entries")

        metric_scales_tensor = torch.tensor(metric_scales, dtype=torch.float32)
        teacher_weights_tensor = torch.tensor(
            teacher_metric_weights, dtype=torch.float32
        )
        pareto_prior_tensor = torch.tensor(pareto_prior, dtype=torch.float32)
        if (metric_scales_tensor <= 0).any():
            raise ValueError("metric scales must be positive")
        if (teacher_weights_tensor < 0).any() or teacher_weights_tensor.sum() <= 0:
            raise ValueError("teacher metric weights must be non-negative and non-zero")
        if (pareto_prior_tensor <= 0).any():
            raise ValueError("pareto prior entries must be strictly positive")
        self.register_buffer("metric_scales", metric_scales_tensor, persistent=True)
        self.register_buffer(
            "teacher_metric_weights",
            teacher_weights_tensor / teacher_weights_tensor.sum(),
            persistent=True,
        )
        self.register_buffer(
            "pareto_prior",
            pareto_prior_tensor / pareto_prior_tensor.sum(),
            persistent=True,
        )
        self.register_buffer(
            "calibration_scales",
            torch.tensor([10.0, 10.0, math.pi, 10.0, 20.0, 2.0]),
            persistent=True,
        )
        self.register_buffer(
            "risk_lcb_margins",
            torch.tensor(risk_lcb_margins, dtype=torch.float32),
            persistent=True,
        )
        self.register_buffer(
            "risk_teacher_margins",
            torch.tensor(risk_teacher_margins, dtype=torch.float32),
            persistent=True,
        )

        self.plan_query_proj = nn.Linear(self.embed_dims, self.token_dims)
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

        # 6 calibration values + 4 availability flags + history flag + 10 ego status.
        self.calibration_encoder = nn.Sequential(
            nn.Linear(21, self.token_dims),
            nn.GELU(),
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims),
        )
        self.candidate_fusion = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.token_dims * 2, self.token_dims),
        )
        self.candidate_norm = nn.LayerNorm(self.token_dims)

        # Only accuracy and collision require prediction.  Consistency and
        # comfort are exactly computable from causal candidate geometry.
        self.learned_metric_head = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, 4),
        )
        self.pareto_head = nn.Sequential(
            nn.LayerNorm(self.token_dims * 2),
            nn.Linear(self.token_dims * 2, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, len(RCPC_METRIC_NAMES)),
        )
        self.score_residual_head = nn.Sequential(
            nn.LayerNorm(self.token_dims),
            nn.Linear(self.token_dims, self.token_dims),
            nn.GELU(),
            nn.Linear(self.token_dims, 1),
        )
        # Zero starts preserve checkpoint behavior exactly.  Ranking gradients
        # immediately train these coefficients because analytic costs vary by mode.
        self.analytic_gain = nn.Parameter(torch.zeros(4))

        # C4-B learns whether the fixed critic should be allowed to replace the
        # base winner.  Its features contain only current model predictions and
        # causal state; GT appears exclusively in the takeover label in loss().
        # Four token vectors plus 25 score/geometry/state scalars are consumed.
        self.trust_gate = None
        if self.enable_trust_gate:
            trust_gate_dims = 4 * self.token_dims + 25
            self.trust_gate = nn.Sequential(
                nn.LayerNorm(trust_gate_dims),
                nn.Linear(trust_gate_dims, self.token_dims),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.token_dims, 1),
            )

        # C4-C replaces scalar takeover classification with four paired
        # base-minus-critic improvement distributions: L2@4/5/6s and
        # avoidable collision.  The first four outputs are medians; the last
        # four parameterize a non-negative interval down to a lower quantile.
        # Eight critic metric differences augment the causal C4-B features.
        self.risk_controller = None
        if self.enable_risk_controller:
            risk_feature_dims = 4 * self.token_dims + 25 + 8
            self.risk_controller = nn.Sequential(
                nn.LayerNorm(risk_feature_dims),
                nn.Linear(risk_feature_dims, self.token_dims),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.token_dims, 8),
            )

        # C4-D keeps the complete C4-C path frozen and learns only this
        # residual, set-wise selector. Its incumbent is exactly the already
        # calibrated C4-C decision, never a reconstructed surrogate.
        self.sequential_selector = None
        if self.enable_causal_pareto_selector:
            self.sequential_selector = CausalInnovationParetoSelector(
                token_dims=self.token_dims,
                num_commands=self.num_commands,
                num_modes=self.num_modes,
                horizon_steps=self.horizon_steps,
                num_heads=int(num_heads),
                dropout=float(dropout),
                lcb_margins=selector_lcb_margins,
                calibration_offsets=selector_calibration_offsets,
                teacher_margins=selector_teacher_margins,
                analytic_tolerances=selector_analytic_tolerances,
                utility_weights=selector_utility_weights,
                utility_residual_scale=selector_utility_residual_scale,
                max_mean_displacement=selector_max_mean_displacement,
                max_endpoint_displacement=selector_max_endpoint_displacement,
                score_gap=selector_score_gap,
                require_history=selector_require_history,
                require_calibration=selector_require_calibration,
                enable_direct_takeover_head=selector_enable_direct_takeover_head,
                takeover_threshold=selector_takeover_threshold,
                takeover_utility_scale=selector_takeover_utility_scale,
                enable_margin_calibrator_head=(
                    selector_enable_margin_calibrator_head
                ),
                margin_threshold=selector_margin_threshold,
                margin_evidence_scale=selector_margin_evidence_scale,
                margin_evidence_temperature=(
                    selector_margin_evidence_temperature
                ),
                margin_utility_scale=selector_margin_utility_scale,
            )

        self.refiner_head = None
        if self.enable_refiner:
            self.refiner_head = nn.Sequential(
                nn.LayerNorm(self.token_dims),
                nn.Linear(self.token_dims, self.token_dims),
                nn.GELU(),
                nn.Linear(self.token_dims, self.horizon_steps * 2),
            )
        self.init_weight()

    def init_weight(self) -> None:
        nn.init.zeros_(self.learned_metric_head[-1].weight)
        nn.init.zeros_(self.learned_metric_head[-1].bias)
        nn.init.zeros_(self.pareto_head[-1].weight)
        nn.init.zeros_(self.pareto_head[-1].bias)
        nn.init.zeros_(self.score_residual_head[-1].weight)
        nn.init.zeros_(self.score_residual_head[-1].bias)
        if self.trust_gate is not None:
            nn.init.zeros_(self.trust_gate[-1].weight)
            # Conservative initialization: abstain until supervised evidence
            # supports a critic takeover.
            nn.init.constant_(self.trust_gate[-1].bias, -2.0)
        if self.risk_controller is not None:
            nn.init.zeros_(self.risk_controller[-1].weight)
            nn.init.zeros_(self.risk_controller[-1].bias)
        if self.refiner_head is not None:
            nn.init.zeros_(self.refiner_head[-1].weight)
            nn.init.zeros_(self.refiner_head[-1].bias)
        with torch.no_grad():
            self.analytic_gain.zero_()

    def _validate_forward_shapes(
        self,
        base_logits: torch.Tensor,
        plan_reg: torch.Tensor,
        plan_query: torch.Tensor,
    ) -> int:
        batch_size = plan_reg.shape[0]
        if tuple(base_logits.shape) != (batch_size, 1, self.num_candidates):
            raise ValueError(
                "base_logits must be [B,1,18], got " + str(tuple(base_logits.shape))
            )
        if tuple(plan_reg.shape) != (
            batch_size,
            1,
            self.num_candidates,
            self.horizon_steps,
            2,
        ):
            raise ValueError(
                "plan_reg must be [B,1,18,12,2], got " + str(tuple(plan_reg.shape))
            )
        if tuple(plan_query.shape) != (
            batch_size,
            1,
            self.num_candidates,
            self.embed_dims,
        ):
            raise ValueError(
                "plan_query must be [B,1,18,D], got " + str(tuple(plan_query.shape))
            )
        return batch_size

    def _trajectory_tokens(self, cumulative: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, num_candidates, horizon, _ = cumulative.shape
        increments = cumulative_to_increments(cumulative)
        velocity = increments / self.dt
        acceleration = torch.zeros_like(velocity)
        acceleration[..., 1:, :] = (
            velocity[..., 1:, :] - velocity[..., :-1, :]
        ) / self.dt
        speed = torch.linalg.norm(velocity, dim=-1, keepdim=True)
        accel = torch.linalg.norm(acceleration, dim=-1, keepdim=True)
        yaw = torch.atan2(increments[..., 1:2], increments[..., 0:1])
        horizon_value = torch.linspace(
            0.0,
            1.0,
            self.horizon_steps,
            device=cumulative.device,
            dtype=cumulative.dtype,
        ).view(1, 1, horizon, 1)
        horizon_value = horizon_value.expand(batch_size, num_candidates, -1, -1)
        geometry = torch.cat(
            [
                cumulative,
                increments,
                speed,
                accel,
                torch.sin(yaw),
                torch.cos(yaw),
                horizon_value,
            ],
            dim=-1,
        )
        tokens = self.geometry_encoder(geometry)
        horizon_index = torch.arange(horizon, device=cumulative.device)
        tokens = tokens + self.horizon_embedding(horizon_index).view(
            1, 1, horizon, self.token_dims
        )
        flat = tokens.reshape(batch_size * num_candidates, horizon, self.token_dims)
        attended, _ = self.horizon_attention(flat, flat, flat, need_weights=False)
        flat = self.horizon_norm(flat + attended)
        tokens = flat.reshape(batch_size, num_candidates, horizon, self.token_dims)
        summary = 0.5 * (tokens.mean(dim=2) + tokens[:, :, -1])
        return tokens, summary

    def _calibration_context(
        self,
        reference: torch.Tensor,
        calibration_features: Optional[torch.Tensor],
        calibration_availability: Optional[torch.Tensor],
        history_valid: Optional[torch.Tensor],
        ego_status: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = reference.shape[0]
        if calibration_features is None:
            calibration_features = reference.new_zeros(batch_size, 6)
        if calibration_availability is None:
            calibration_availability = reference.new_zeros(batch_size, 4)
        if history_valid is None:
            history_valid = torch.zeros(
                batch_size, device=reference.device, dtype=torch.bool
            )
        calibration_features = calibration_features.reshape(batch_size, 6).to(
            device=reference.device, dtype=reference.dtype
        )
        calibration_availability = calibration_availability.reshape(batch_size, 4).to(
            device=reference.device, dtype=reference.dtype
        )
        scales = self.calibration_scales.to(
            device=reference.device, dtype=reference.dtype
        )
        calibration_norm = torch.clamp(
            calibration_features.abs() / scales, 0.0, 5.0
        )
        calibration_norm[:, :4] = (
            calibration_norm[:, :4] * calibration_availability
        )
        history_value = history_valid.reshape(batch_size, 1).to(
            device=reference.device, dtype=reference.dtype
        )
        if ego_status is None:
            ego = reference.new_zeros(batch_size, 10)
        else:
            ego = ego_status.to(device=reference.device, dtype=reference.dtype)
            ego = ego.reshape(batch_size, -1)
            if ego.shape[-1] < 10:
                ego = F.pad(ego, (0, 10 - ego.shape[-1]))
            ego = ego[:, :10]
        ego = torch.clamp(ego / 20.0, -5.0, 5.0)
        return self.calibration_encoder(
            torch.cat(
                [calibration_norm, calibration_availability, history_value, ego],
                dim=-1,
            )
        )

    def _analytic_metrics(
        self,
        cumulative: torch.Tensor,
        last_final_planning: Optional[torch.Tensor],
        history_valid: Optional[torch.Tensor],
    ) -> torch.Tensor:
        batch_size = cumulative.shape[0]
        if last_final_planning is None:
            last_final_planning = cumulative.new_zeros(
                batch_size, self.horizon_steps, 2
            )
        if history_valid is None:
            history_valid = torch.zeros(
                batch_size, device=cumulative.device, dtype=torch.bool
            )
        last_final_planning = last_final_planning.to(
            device=cumulative.device, dtype=cumulative.dtype
        )
        consistency_distance = torch.linalg.norm(
            cumulative - last_final_planning[:, None], dim=-1
        )
        history_mask = history_valid.reshape(batch_size, 1).to(cumulative.dtype)
        consistency = []
        for metric_index, horizon in zip((4, 5, 6), (8, 10, 12)):
            value = consistency_distance[..., :horizon].mean(dim=-1)
            scale = self.metric_scales[metric_index].to(
                device=cumulative.device, dtype=cumulative.dtype
            )
            consistency.append(torch.clamp(value / scale, 0.0, 5.0) * history_mask)

        increments = cumulative_to_increments(cumulative)
        velocity = increments / self.dt
        acceleration = torch.zeros_like(velocity)
        acceleration[..., 1:, :] = (
            velocity[..., 1:, :] - velocity[..., :-1, :]
        ) / self.dt
        jerk = torch.zeros_like(acceleration)
        jerk[..., 1:, :] = (
            acceleration[..., 1:, :] - acceleration[..., :-1, :]
        ) / self.dt
        comfort = torch.linalg.norm(jerk, dim=-1).mean(dim=-1)
        comfort = torch.clamp(
            comfort
            / self.metric_scales[7].to(
                device=cumulative.device, dtype=cumulative.dtype
            ),
            0.0,
            5.0,
        )
        return torch.stack(consistency + [comfort], dim=-1)

    def forward(
        self,
        base_logits: torch.Tensor,
        plan_reg: torch.Tensor,
        plan_query: torch.Tensor,
        agent_features: torch.Tensor,
        map_features: torch.Tensor,
        last_final_planning: Optional[torch.Tensor] = None,
        history_valid: Optional[torch.Tensor] = None,
        calibration_features: Optional[torch.Tensor] = None,
        calibration_availability: Optional[torch.Tensor] = None,
        ego_status: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size = self._validate_forward_shapes(base_logits, plan_reg, plan_query)
        source_plan_reg = plan_reg.detach() if self.detach_inputs else plan_reg
        source_query = plan_query.detach() if self.detach_inputs else plan_query
        source_agents = agent_features.detach() if self.detach_inputs else agent_features
        source_map = map_features.detach() if self.detach_inputs else map_features

        # Critic inputs may be detached during the ranker-only pilot, while the
        # output trajectory must retain the original planning-regression graph.
        original_cumulative = plan_reg.squeeze(1).cumsum(dim=-2)
        base_cumulative = source_plan_reg.squeeze(1).cumsum(dim=-2)
        horizon_tokens, geometry_summary = self._trajectory_tokens(base_cumulative)
        candidate = self.plan_query_proj(source_query.squeeze(1)) + geometry_summary

        scene_tokens = torch.cat(
            [self.agent_proj(source_agents), self.map_proj(source_map)], dim=1
        )
        scene_update, _ = self.scene_attention(
            candidate, scene_tokens, scene_tokens, need_weights=False
        )
        candidate = self.scene_norm(candidate + scene_update)

        calibration_context = self._calibration_context(
            candidate,
            calibration_features,
            calibration_availability,
            history_valid,
            ego_status,
        )
        candidate = candidate + calibration_context[:, None]
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

        learned_metrics = F.softplus(self.learned_metric_head(candidate))
        analytic_metrics = self._analytic_metrics(
            base_cumulative, last_final_planning, history_valid
        )
        analytic_gain = self.analytic_gain.to(dtype=candidate.dtype).view(1, 1, 4)
        effective_metrics = torch.cat(
            [learned_metrics, analytic_metrics * analytic_gain], dim=-1
        )

        command_context = candidate.reshape(
            batch_size, self.num_commands, self.num_modes, self.token_dims
        ).mean(dim=2)
        pareto_input = torch.cat(
            [
                command_context,
                calibration_context[:, None].expand(-1, self.num_commands, -1),
            ],
            dim=-1,
        )
        prior_log = torch.log(self.pareto_prior.clamp_min(1e-8)).to(
            device=candidate.device, dtype=candidate.dtype
        )
        pareto_weights = torch.softmax(
            prior_log.view(1, 1, -1) + self.pareto_head(pareto_input), dim=-1
        )
        metric_cost = (
            effective_metrics.reshape(
                batch_size, self.num_commands, self.num_modes, -1
            )
            * pareto_weights[:, :, None]
        ).sum(dim=-1)
        centered_cost = metric_cost - metric_cost.mean(dim=-1, keepdim=True)
        residual_score = self.score_residual_head(candidate).squeeze(-1).reshape(
            batch_size, self.num_commands, self.num_modes
        )
        raw_delta = (
            -self.score_scale * centered_cost
            + self.residual_score_scale * residual_score
        )
        bounded_delta = self.max_score_delta * torch.tanh(
            raw_delta / max(self.max_score_delta, 1e-6)
        )
        applied_delta = self.score_blend * bounded_delta
        base_by_command = base_logits.reshape(
            batch_size, self.num_commands, self.num_modes
        )
        critic_by_command = base_by_command + applied_delta
        critic_logits = critic_by_command.reshape(
            batch_size, 1, self.num_candidates
        )
        calibrated_logits = critic_logits
        final_applied_delta = applied_delta
        trust_gate_logits = None
        trust_gate_probability = None
        trust_gate_applied = None
        risk_metric_median = None
        risk_metric_spread = None
        risk_metric_lcb = None
        risk_controller_applied = None
        risk_predicted_pareto_safe = None
        risk_structural_safe = None
        risk_analytic_improvement = None
        risk_candidate_displacement = None
        selector_output = None
        base_winner = base_by_command.detach().argmax(dim=-1)
        critic_winner = critic_by_command.detach().argmax(dim=-1)

        if (
            self.trust_gate is not None
            or self.risk_controller is not None
            or self.sequential_selector is not None
        ):
            candidate_by_command = candidate.reshape(
                batch_size,
                self.num_commands,
                self.num_modes,
                self.token_dims,
            )
            trajectory_by_command = base_cumulative.reshape(
                batch_size,
                self.num_commands,
                self.num_modes,
                self.horizon_steps,
                2,
            )

            def gather_mode(value: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
                extra_shape = value.shape[3:]
                gather_index = index[..., None]
                for _ in extra_shape:
                    gather_index = gather_index.unsqueeze(-1)
                gather_index = gather_index.expand(
                    batch_size, self.num_commands, 1, *extra_shape
                )
                return torch.gather(value, 2, gather_index).squeeze(2)

            base_embedding = gather_mode(candidate_by_command, base_winner)
            critic_embedding = gather_mode(candidate_by_command, critic_winner)
            base_trajectory = gather_mode(trajectory_by_command, base_winner)
            critic_trajectory = gather_mode(trajectory_by_command, critic_winner)
            displacement = torch.linalg.norm(
                critic_trajectory - base_trajectory, dim=-1
            )

            base_top2 = torch.topk(
                base_by_command.detach(), 2, dim=-1, largest=True, sorted=True
            ).values
            critic_top2 = torch.topk(
                critic_by_command.detach(), 2, dim=-1, largest=True, sorted=True
            ).values
            base_metric_cost = torch.gather(
                metric_cost, 2, base_winner[..., None]
            ).squeeze(-1)
            critic_metric_cost = torch.gather(
                metric_cost, 2, critic_winner[..., None]
            ).squeeze(-1)
            if history_valid is None:
                history_feature = base_by_command.new_zeros(
                    batch_size, self.num_commands, 1
                )
            else:
                history_feature = history_valid.reshape(batch_size, 1, 1).to(
                    device=base_by_command.device, dtype=base_by_command.dtype
                ).expand(-1, self.num_commands, -1)
            if calibration_availability is None:
                calibration_feature = base_by_command.new_zeros(
                    batch_size, self.num_commands, 1
                )
            else:
                calibration_feature = (
                    calibration_availability.reshape(batch_size, 4)
                    .to(device=base_by_command.device)
                    .bool()
                    .any(dim=-1)
                    .reshape(batch_size, 1, 1)
                    .to(dtype=base_by_command.dtype)
                    .expand(-1, self.num_commands, -1)
                )
            scalar_features = torch.cat(
                [
                    base_by_command - base_by_command.mean(dim=-1, keepdim=True),
                    applied_delta - applied_delta.mean(dim=-1, keepdim=True),
                    metric_cost - metric_cost.mean(dim=-1, keepdim=True),
                    (base_top2[..., :1] - base_top2[..., 1:2]),
                    (critic_top2[..., :1] - critic_top2[..., 1:2]),
                    (base_metric_cost - critic_metric_cost).unsqueeze(-1),
                    displacement.mean(dim=-1, keepdim=True),
                    displacement[..., -1:].contiguous(),
                    history_feature,
                    calibration_feature,
                ],
                dim=-1,
            )
            arbitration_features = torch.cat(
                [
                    command_context,
                    base_embedding,
                    critic_embedding - base_embedding,
                    calibration_context[:, None].expand(
                        -1, self.num_commands, -1
                    ),
                    scalar_features,
                ],
                dim=-1,
            ).detach()
            if self.trust_gate is not None:
                trust_gate_logits = self.trust_gate(
                    arbitration_features
                ).squeeze(-1)
                trust_gate_probability = torch.sigmoid(trust_gate_logits)
                trust_gate_applied = (
                    trust_gate_probability >= self.trust_gate_threshold
                ) & (critic_winner != base_winner)
                calibrated_by_command = torch.where(
                    trust_gate_applied[..., None],
                    critic_by_command,
                    base_by_command,
                )
                calibrated_logits = calibrated_by_command.reshape(
                    batch_size, 1, self.num_candidates
                )
                final_applied_delta = torch.where(
                    trust_gate_applied[..., None],
                    applied_delta,
                    torch.zeros_like(applied_delta),
                )

            if self.risk_controller is not None:
                effective_by_command = effective_metrics.reshape(
                    batch_size,
                    self.num_commands,
                    self.num_modes,
                    len(RCPC_METRIC_NAMES),
                )
                base_effective_metric = gather_mode(
                    effective_by_command, base_winner
                )
                critic_effective_metric = gather_mode(
                    effective_by_command, critic_winner
                )
                predicted_metric_improvement = (
                    base_effective_metric - critic_effective_metric
                )
                risk_features = torch.cat(
                    [arbitration_features, predicted_metric_improvement.detach()],
                    dim=-1,
                )
                risk_raw = self.risk_controller(risk_features)
                risk_metric_median = risk_raw[..., :4]
                risk_metric_spread = F.softplus(risk_raw[..., 4:])
                risk_metric_lcb = risk_metric_median - risk_metric_spread

                analytic_by_command = analytic_metrics.reshape(
                    batch_size,
                    self.num_commands,
                    self.num_modes,
                    4,
                )
                risk_analytic_improvement = gather_mode(
                    analytic_by_command, base_winner
                ) - gather_mode(analytic_by_command, critic_winner)
                risk_candidate_displacement = displacement
                lcb_margins = self.risk_lcb_margins.to(
                    device=risk_metric_lcb.device,
                    dtype=risk_metric_lcb.dtype,
                ).view(1, 1, 4)
                lcb_margins = lcb_margins + risk_metric_lcb.new_tensor(
                    self.risk_lcb_calibration_offsets
                ).view(1, 1, 4)
                risk_predicted_pareto_safe = (
                    risk_metric_lcb >= lcb_margins
                ).all(dim=-1)
                risk_structural_safe = (
                    risk_analytic_improvement
                    >= -self.risk_analytic_tolerance
                ).all(dim=-1)
                risk_structural_safe = risk_structural_safe & (
                    displacement.mean(dim=-1)
                    <= self.risk_max_mean_displacement
                )
                risk_structural_safe = risk_structural_safe & (
                    displacement[..., -1]
                    <= self.risk_max_endpoint_displacement
                )
                risk_controller_applied = (
                    risk_predicted_pareto_safe
                    & risk_structural_safe
                    & (critic_winner != base_winner)
                )
                calibrated_by_command = torch.where(
                    risk_controller_applied[..., None],
                    critic_by_command,
                    base_by_command,
                )
                calibrated_logits = calibrated_by_command.reshape(
                    batch_size, 1, self.num_candidates
                )
                final_applied_delta = torch.where(
                    risk_controller_applied[..., None],
                    applied_delta,
                    torch.zeros_like(applied_delta),
                )

            if self.sequential_selector is not None:
                if self.risk_controller is None:
                    raise RuntimeError("C4-D requires an active C4-C incumbent")
                selector_output = self.sequential_selector(
                    candidate_tokens=candidate_by_command.detach(),
                    command_context=command_context.detach(),
                    calibration_context=calibration_context.detach(),
                    trajectories=trajectory_by_command.detach(),
                    incumbent_logits=calibrated_by_command.detach(),
                    effective_metrics=effective_by_command.detach(),
                    analytic_metrics=analytic_by_command.detach(),
                    calibration_features=calibration_features,
                    calibration_availability=calibration_availability,
                    history_valid=history_valid,
                )
                calibrated_by_command = selector_output["calibrated_logits"]
                calibrated_logits = calibrated_by_command.reshape(
                    batch_size, 1, self.num_candidates
                )
                final_applied_delta = calibrated_by_command - base_by_command

        raw_refined_cumulative = original_cumulative
        residual_cumulative = torch.zeros_like(base_cumulative)
        top_m_mask = torch.ones(
            batch_size,
            self.num_commands,
            self.num_modes,
            device=base_logits.device,
            dtype=torch.bool,
        )
        refined_cumulative = original_cumulative
        if self.refiner_head is not None:
            raw_residual = self.refiner_head(candidate).reshape(
                batch_size,
                self.num_candidates,
                self.horizon_steps,
                2,
            )
            horizon_scale = torch.linspace(
                0.25,
                self.max_refine_offset,
                self.horizon_steps,
                device=raw_residual.device,
                dtype=raw_residual.dtype,
            ).view(1, 1, self.horizon_steps, 1)
            residual_cumulative = torch.tanh(raw_residual) * horizon_scale
            raw_refined_cumulative = original_cumulative + residual_cumulative
            score_by_command = calibrated_logits.reshape(
                batch_size, self.num_commands, self.num_modes
            ).detach()
            top_indices = torch.topk(
                score_by_command, self.top_m, dim=-1, largest=True, sorted=False
            ).indices
            top_m_mask = torch.zeros_like(score_by_command, dtype=torch.bool)
            top_m_mask.scatter_(-1, top_indices, True)
            flat_mask = top_m_mask.reshape(batch_size, self.num_candidates, 1, 1)
            refined_cumulative = torch.where(
                flat_mask, raw_refined_cumulative, original_cumulative
            )

        if self.refiner_head is None:
            # Preserve the loaded checkpoint bit-for-bit in the ranker pilot.
            refined_plan_reg = plan_reg
        else:
            applied_residual = refined_cumulative - original_cumulative
            refined_plan_reg = plan_reg + cumulative_to_increments(
                applied_residual
            ).unsqueeze(1)
        output = {
            "calibrated_logits": calibrated_logits,
            "base_logits": base_logits,
            "critic_logits": critic_logits,
            "refined_plan_reg": refined_plan_reg,
            "score_delta": final_applied_delta.reshape(
                batch_size, 1, self.num_candidates
            ),
            "proposed_score_delta": applied_delta.reshape(
                batch_size, 1, self.num_candidates
            ),
            "unblended_score_delta": bounded_delta.reshape(
                batch_size, 1, self.num_candidates
            ),
            "learned_metrics": learned_metrics,
            "analytic_metrics": analytic_metrics,
            "effective_metrics": effective_metrics,
            "pareto_weights": pareto_weights,
            "base_cumulative": base_cumulative,
            "raw_refined_cumulative": raw_refined_cumulative,
            "refined_cumulative": refined_cumulative,
            "residual_cumulative": residual_cumulative,
            "top_m_mask": top_m_mask,
            "history_reference": (
                base_cumulative.new_zeros(batch_size, self.horizon_steps, 2)
                if last_final_planning is None
                else last_final_planning.detach().to(
                    device=base_cumulative.device, dtype=base_cumulative.dtype
                )
            ),
            "history_valid": (
                torch.zeros(batch_size, device=base_cumulative.device, dtype=torch.bool)
                if history_valid is None
                else history_valid.detach().bool()
            ),
            "calibration_available": (
                torch.zeros(batch_size, device=base_cumulative.device, dtype=torch.bool)
                if calibration_availability is None
                else calibration_availability.detach().to(
                    device=base_cumulative.device
                ).reshape(batch_size, 4).any(dim=-1)
            ),
        }
        if trust_gate_logits is not None:
            output.update(
                {
                    "trust_gate_logits": trust_gate_logits,
                    "trust_gate_probability": trust_gate_probability,
                    "trust_gate_applied": trust_gate_applied,
                    "base_winner": base_winner,
                    "critic_winner": critic_winner,
                }
            )
        if risk_metric_lcb is not None:
            output.update(
                {
                    "risk_metric_median": risk_metric_median,
                    "risk_metric_spread": risk_metric_spread,
                    "risk_metric_lcb": risk_metric_lcb,
                    "risk_controller_applied": risk_controller_applied,
                    "risk_predicted_pareto_safe": risk_predicted_pareto_safe,
                    "risk_structural_safe": risk_structural_safe,
                    "risk_analytic_improvement": risk_analytic_improvement,
                    "risk_candidate_displacement": risk_candidate_displacement,
                    "base_winner": base_winner,
                    "critic_winner": critic_winner,
                }
            )
        if selector_output is not None:
            output.update(
                {
                    "selector_metric_median": selector_output["metric_median"],
                    "selector_metric_spread": selector_output["metric_spread"],
                    "selector_metric_lcb": selector_output["metric_lcb"],
                    "selector_predicted_pareto_safe": selector_output[
                        "predicted_safe"
                    ],
                    "selector_structural_safe": selector_output[
                        "structural_safe"
                    ],
                    "selector_observability_safe": selector_output[
                        "observability_safe"
                    ],
                    "selector_eligible": selector_output["eligible"],
                    "selector_applied": selector_output["selector_applied"],
                    "selector_incumbent_winner": selector_output[
                        "incumbent_winner"
                    ],
                    "selector_winner": selector_output["selector_winner"],
                    "selector_analytic_improvement": selector_output[
                        "analytic_improvement"
                    ],
                    "selector_candidate_displacement": selector_output[
                        "candidate_displacement"
                    ],
                    "selector_candidate_utility": selector_output[
                        "candidate_utility"
                    ],
                    "selector_utility_residual": selector_output[
                        "utility_residual"
                    ],
                    "selector_innovation_gate": selector_output[
                        "innovation_gate"
                    ],
                }
            )
            if "takeover_logit" in selector_output:
                output.update(
                    {
                        "selector_takeover_logit": selector_output[
                            "takeover_logit"
                        ],
                        "selector_takeover_probability": selector_output[
                            "takeover_probability"
                        ],
                        "selector_takeover_proposal_score": selector_output[
                            "takeover_proposal_score"
                        ],
                        "selector_direct_eligible": selector_output[
                            "direct_eligible"
                        ],
                        "selector_direct_proposed_winner": selector_output[
                            "direct_proposed_winner"
                        ],
                        "selector_proposed_winner": selector_output[
                            "proposed_winner"
                        ],
                    }
                )
            if "margin_prediction" in selector_output:
                output.update(
                    {
                        "selector_margin_prediction": selector_output[
                            "margin_prediction"
                        ],
                        "selector_margin_joint_logit": selector_output[
                            "margin_joint_logit"
                        ],
                        "selector_margin_evidence": selector_output[
                            "margin_evidence"
                        ],
                        "selector_margin_safety_score": selector_output[
                            "margin_safety_score"
                        ],
                        "selector_margin_probability": selector_output[
                            "margin_probability"
                        ],
                    }
                )
        for name, value in output.items():
            if torch.is_tensor(value) and value.is_floating_point():
                if not torch.isfinite(value).all():
                    raise FloatingPointError("RCPC output %s contains NaN/Inf" % name)
        return output

    @torch.no_grad()
    def build_targets(
        self,
        output: Dict[str, torch.Tensor],
        data: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        base = output["base_cumulative"].detach()
        batch_size = base.shape[0]
        command = data["gt_ego_fut_cmd"].argmax(dim=-1).long()
        batch_index = torch.arange(batch_size, device=base.device)
        candidate = base.reshape(
            batch_size,
            self.num_commands,
            self.num_modes,
            self.horizon_steps,
            2,
        )[batch_index, command]
        gt_increment = data["gt_ego_fut_trajs"].to(
            device=base.device, dtype=base.dtype
        )
        gt_mask = data["gt_ego_fut_masks"].to(device=base.device).bool()
        gt_cumulative = gt_increment.cumsum(dim=-2)
        distance = torch.linalg.norm(candidate - gt_cumulative[:, None], dim=-1)

        target = base.new_zeros(batch_size, self.num_modes, len(RCPC_METRIC_NAMES))
        valid = torch.zeros_like(target, dtype=torch.bool)
        for metric_index, horizon in enumerate((8, 10, 12)):
            value, horizon_valid = _masked_horizon_mean(
                distance, gt_mask[:, None].expand(-1, self.num_modes, -1), horizon
            )
            scale = self.metric_scales[metric_index].to(
                device=base.device, dtype=base.dtype
            )
            target[..., metric_index] = torch.clamp(value / scale, 0.0, 5.0)
            valid[..., metric_index] = horizon_valid

        collision_available: List[bool] = []
        for batch_id in range(batch_size):
            collision, available = avoidable_collision_proxy(
                candidate[batch_id],
                gt_cumulative[batch_id],
                gt_mask[batch_id],
                _batch_item(data.get("gt_bboxes_3d"), batch_id, batch_size),
                _batch_item(data.get("gt_agent_fut_trajs"), batch_id, batch_size),
                _batch_item(data.get("gt_agent_fut_masks"), batch_id, batch_size),
            )
            target[batch_id, :, 3] = collision
            valid[batch_id, :, 3] = bool(available)
            collision_available.append(bool(available))

        history_reference = output["history_reference"].detach()
        history_valid = output["history_valid"].detach().bool()
        consistency_distance = torch.linalg.norm(
            candidate - history_reference[:, None], dim=-1
        )
        for metric_index, horizon in zip((4, 5, 6), (8, 10, 12)):
            value, horizon_valid = _masked_horizon_mean(
                consistency_distance,
                gt_mask[:, None].expand(-1, self.num_modes, -1),
                horizon,
            )
            scale = self.metric_scales[metric_index].to(
                device=base.device, dtype=base.dtype
            )
            target[..., metric_index] = torch.clamp(value / scale, 0.0, 5.0)
            valid[..., metric_index] = horizon_valid & history_valid[:, None]

        analytic = output["analytic_metrics"].detach().reshape(
            batch_size, self.num_commands, self.num_modes, 4
        )[batch_index, command]
        target[..., 7] = analytic[..., 3]
        valid[..., 7] = True

        teacher_weight = self.teacher_metric_weights.to(
            device=base.device, dtype=base.dtype
        ).view(1, 1, -1)
        effective_weight = teacher_weight * valid.to(base.dtype)
        effective_weight = effective_weight / effective_weight.sum(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        teacher_cost = (target * effective_weight).sum(dim=-1)
        rank_valid = valid[..., :3].any(dim=-1).any(dim=-1)
        return {
            "metric_target": target,
            "metric_valid": valid,
            "teacher_cost": teacher_cost,
            "rank_valid": rank_valid,
            "command": command,
            "gt_cumulative": gt_cumulative,
            "gt_mask": gt_mask,
            "collision_available": torch.tensor(
                collision_available, device=base.device, dtype=torch.bool
            ),
        }

    def _score_spearman(
        self, score: torch.Tensor, teacher_cost: torch.Tensor
    ) -> torch.Tensor:
        score_rank = torch.argsort(torch.argsort(score, dim=-1), dim=-1).float()
        quality_rank = torch.argsort(
            torch.argsort(-teacher_cost, dim=-1), dim=-1
        ).float()
        score_rank = score_rank - score_rank.mean(dim=-1, keepdim=True)
        quality_rank = quality_rank - quality_rank.mean(dim=-1, keepdim=True)
        numerator = (score_rank * quality_rank).sum(dim=-1)
        denominator = torch.sqrt(
            score_rank.square().sum(dim=-1)
            * quality_rank.square().sum(dim=-1)
        ).clamp_min(1e-6)
        return numerator / denominator

    def loss(
        self,
        output: Dict[str, torch.Tensor],
        data: Dict[str, object],
    ) -> Dict[str, torch.Tensor]:
        targets = self.build_targets(output, data)
        base = output["base_cumulative"]
        batch_size = base.shape[0]
        batch_index = torch.arange(batch_size, device=base.device)
        command = targets["command"]
        rank_valid = targets["rank_valid"]

        learned_metric = output["learned_metrics"].reshape(
            batch_size, self.num_commands, self.num_modes, 4
        )[batch_index, command]
        metric_target = targets["metric_target"][..., :4]
        metric_valid = targets["metric_valid"][..., :4].to(base.dtype)
        metric_denom = metric_valid.sum().clamp_min(1.0)
        metric_loss = (
            F.smooth_l1_loss(
                learned_metric, metric_target, reduction="none", beta=0.2
            )
            * metric_valid
        ).sum() / metric_denom

        # The critic remains a dense ranker even when the trust gate abstains.
        # This prevents hard arbitration from starving its listwise gradients.
        score = output.get("critic_logits", output["calibrated_logits"]).reshape(
            batch_size, self.num_commands, self.num_modes
        )[batch_index, command]
        teacher_cost = targets["teacher_cost"]
        teacher_probability = torch.softmax(
            -teacher_cost / self.listwise_temperature, dim=-1
        )
        listwise_per_sample = -(
            teacher_probability * F.log_softmax(score, dim=-1)
        ).sum(dim=-1)
        valid_float = rank_valid.to(base.dtype)
        listwise_loss = (listwise_per_sample * valid_float).sum() / valid_float.sum().clamp_min(1.0)

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
            & rank_valid[:, None, None]
        )
        pair_loss_value = F.softplus(-preference * score_difference)
        pairwise_loss = (
            pair_loss_value * pair_mask.to(base.dtype)
        ).sum() / pair_mask.to(base.dtype).sum().clamp_min(1.0)

        pareto_weight = output["pareto_weights"][batch_index, command]
        prior = self.pareto_prior.to(
            device=base.device, dtype=base.dtype
        ).view(1, -1)
        pareto_prior_loss = (
            pareto_weight
            * (
                torch.log(pareto_weight.clamp_min(1e-8))
                - torch.log(prior.clamp_min(1e-8))
            )
        ).sum(dim=-1).mean()

        losses = {
            "rcpc_loss_metric": metric_loss * self.metric_loss_weight,
            "rcpc_loss_listwise": listwise_loss * self.listwise_loss_weight,
            "rcpc_loss_pairwise": pairwise_loss * self.pairwise_loss_weight,
            "rcpc_loss_pareto_prior": (
                pareto_prior_loss * self.pareto_prior_loss_weight
            ),
        }

        gate_diagnostics = None
        if self.trust_gate is not None:
            gate_logits = output["trust_gate_logits"][batch_index, command]
            base_score = output["base_logits"].reshape(
                batch_size, self.num_commands, self.num_modes
            )[batch_index, command]
            proposed_score = output["critic_logits"].reshape(
                batch_size, self.num_commands, self.num_modes
            )[batch_index, command]
            base_selected = base_score.detach().argmax(dim=-1)
            proposed_selected = proposed_score.detach().argmax(dim=-1)
            base_teacher_cost = teacher_cost.gather(
                1, base_selected[:, None]
            ).squeeze(1)
            proposed_teacher_cost = teacher_cost.gather(
                1, proposed_selected[:, None]
            ).squeeze(1)
            teacher_improvement = base_teacher_cost - proposed_teacher_cost
            proposal_changes_mode = proposed_selected != base_selected
            gate_target = proposal_changes_mode & (
                teacher_improvement > self.trust_gate_teacher_margin
            )
            gate_target_float = gate_target.to(base.dtype)
            gate_valid_float = rank_valid.to(base.dtype)
            gate_weight = 1.0 + (
                self.trust_gate_positive_weight - 1.0
            ) * gate_target_float
            gate_bce = F.binary_cross_entropy_with_logits(
                gate_logits, gate_target_float, reduction="none"
            )
            gate_loss = (
                gate_bce * gate_weight * gate_valid_float
            ).sum() / (gate_weight * gate_valid_float).sum().clamp_min(1.0)
            losses["rcpc_loss_trust_gate"] = (
                gate_loss * self.trust_gate_loss_weight
            )
            gate_diagnostics = {
                "logits": gate_logits,
                "target": gate_target,
                "changed": proposal_changes_mode,
                "teacher_improvement": teacher_improvement,
            }

        risk_diagnostics = None
        if self.risk_controller is not None:
            base_selected = output["base_winner"][batch_index, command]
            critic_selected = output["critic_winner"][batch_index, command]
            metric_target = targets["metric_target"]
            metric_valid_bool = targets["metric_valid"]

            def gather_target(value, index):
                gather_index = index[:, None, None].expand(
                    batch_size, 1, value.shape[-1]
                )
                return torch.gather(value, 1, gather_index).squeeze(1)

            base_metric_target = gather_target(metric_target, base_selected)
            critic_metric_target = gather_target(metric_target, critic_selected)
            base_metric_valid = gather_target(
                metric_valid_bool, base_selected
            ).bool()
            critic_metric_valid = gather_target(
                metric_valid_bool, critic_selected
            ).bool()
            paired_metric_valid = (
                base_metric_valid
                & critic_metric_valid
                & rank_valid[:, None]
            )
            metric_improvement_target = (
                base_metric_target - critic_metric_target
            )
            risk_median = output["risk_metric_median"][batch_index, command]
            risk_lcb = output["risk_metric_lcb"][batch_index, command]
            learned_valid = paired_metric_valid[..., :4].to(base.dtype)
            learned_denom = learned_valid.sum().clamp_min(1.0)

            median_error = F.smooth_l1_loss(
                risk_median,
                metric_improvement_target[..., :4],
                reduction="none",
                beta=0.05,
            )
            median_loss = (median_error * learned_valid).sum() / learned_denom
            quantile_residual = (
                metric_improvement_target[..., :4] - risk_lcb
            )
            quantile_loss_value = torch.maximum(
                self.risk_quantile * quantile_residual,
                (self.risk_quantile - 1.0) * quantile_residual,
            )
            quantile_loss = (
                quantile_loss_value * learned_valid
            ).sum() / learned_denom
            losses["rcpc_loss_risk_median"] = (
                median_loss * self.risk_median_loss_weight
            )
            losses["rcpc_loss_risk_quantile"] = (
                quantile_loss * self.risk_quantile_loss_weight
            )

            lcb_margin = self.risk_lcb_margins.to(
                device=base.device, dtype=base.dtype
            ).view(1, 4)
            lcb_margin = lcb_margin + risk_lcb.new_tensor(
                self.risk_lcb_calibration_offsets
            ).view(1, 4)
            teacher_margin = self.risk_teacher_margins.to(
                device=base.device, dtype=base.dtype
            ).view(1, 4)
            risk_decision_logit = (
                (risk_lcb - lcb_margin).min(dim=-1).values
                / self.risk_decision_temperature
            )
            proposal_changes_mode = critic_selected != base_selected
            learned_teacher_safe = (
                metric_improvement_target[..., :4] >= teacher_margin
            ).all(dim=-1) & paired_metric_valid[..., :4].all(dim=-1)
            analytic_valid_or_safe = (
                (~paired_metric_valid[..., 4:])
                | (
                    metric_improvement_target[..., 4:]
                    >= -self.risk_analytic_tolerance
                )
            ).all(dim=-1)
            risk_teacher_safe = (
                proposal_changes_mode
                & learned_teacher_safe
                & analytic_valid_or_safe
                & rank_valid
            )
            decision_valid = proposal_changes_mode & rank_valid
            decision_weight = torch.where(
                risk_teacher_safe,
                risk_decision_logit.new_ones(()),
                risk_decision_logit.new_full(
                    (), self.risk_false_takeover_weight
                ),
            )
            decision_bce = F.binary_cross_entropy_with_logits(
                risk_decision_logit,
                risk_teacher_safe.to(base.dtype),
                reduction="none",
            )
            risk_decision_loss = (
                decision_bce
                * decision_weight
                * decision_valid.to(base.dtype)
            ).sum() / (
                decision_weight * decision_valid.to(base.dtype)
            ).sum().clamp_min(1.0)
            losses["rcpc_loss_risk_decision"] = (
                risk_decision_loss * self.risk_decision_loss_weight
            )
            risk_diagnostics = {
                "metric_improvement_target": metric_improvement_target,
                "paired_metric_valid": paired_metric_valid,
                "lcb": risk_lcb,
                "teacher_safe": risk_teacher_safe,
                "changed": proposal_changes_mode,
                "decision_logit": risk_decision_logit,
            }

        selector_diagnostics = None
        if self.sequential_selector is not None:
            metric_target = targets["metric_target"]
            metric_valid_bool = targets["metric_valid"].bool()
            incumbent_selected = output["selector_incumbent_winner"][
                batch_index, command
            ]

            def gather_selector_target(value, index):
                gather_index = index[:, None, None].expand(
                    batch_size, 1, value.shape[-1]
                )
                return torch.gather(value, 1, gather_index).squeeze(1)

            incumbent_metric_target = gather_selector_target(
                metric_target, incumbent_selected
            )
            incumbent_metric_valid = gather_selector_target(
                metric_valid_bool, incumbent_selected
            ).bool()
            selector_metric_improvement_target = (
                incumbent_metric_target[:, None] - metric_target
            )
            selector_paired_valid = (
                incumbent_metric_valid[:, None]
                & metric_valid_bool
                & rank_valid[:, None, None]
            )
            selector_median = output["selector_metric_median"][
                batch_index, command
            ]
            selector_lcb = output["selector_metric_lcb"][
                batch_index, command
            ]
            selector_learned_valid = selector_paired_valid[..., :4].to(
                base.dtype
            )
            selector_learned_denom = selector_learned_valid.sum().clamp_min(1.0)

            selector_median_error = F.smooth_l1_loss(
                selector_median,
                selector_metric_improvement_target[..., :4],
                reduction="none",
                beta=0.05,
            )
            selector_median_loss = (
                selector_median_error * selector_learned_valid
            ).sum() / selector_learned_denom
            selector_quantile_residual = (
                selector_metric_improvement_target[..., :4] - selector_lcb
            )
            selector_quantile_value = torch.maximum(
                self.selector_quantile * selector_quantile_residual,
                (self.selector_quantile - 1.0) * selector_quantile_residual,
            )
            selector_quantile_loss = (
                selector_quantile_value * selector_learned_valid
            ).sum() / selector_learned_denom
            losses["rcpc_loss_selector_median"] = (
                selector_median_loss * self.selector_median_loss_weight
            )
            losses["rcpc_loss_selector_quantile"] = (
                selector_quantile_loss * self.selector_quantile_loss_weight
            )

            selector_margin = self.sequential_selector.lcb_margins.to(
                device=base.device, dtype=base.dtype
            ) + selector_lcb.new_tensor(
                self.sequential_selector.calibration_offsets
            )
            selector_margin = selector_margin.view(1, 1, 4)
            selector_teacher_margin = self.sequential_selector.teacher_margins.to(
                device=base.device, dtype=base.dtype
            ).view(1, 1, 4)
            selector_analytic_tolerance = (
                self.sequential_selector.analytic_tolerances.to(
                    device=base.device, dtype=base.dtype
                ).view(1, 1, 4)
            )
            selector_decision_logit = (
                (selector_lcb - selector_margin).min(dim=-1).values
                / self.selector_decision_temperature
            )
            mode_index = torch.arange(
                self.num_modes, device=base.device
            ).view(1, self.num_modes)
            selector_changed = mode_index != incumbent_selected[:, None]
            selector_observable = output["selector_observability_safe"][
                batch_index, command
            ]
            selector_learned_safe = (
                selector_metric_improvement_target[..., :4]
                >= selector_teacher_margin
            ).all(dim=-1) & selector_paired_valid[..., :4].all(dim=-1)
            selector_analytic_safe = (
                selector_metric_improvement_target[..., 4:]
                >= -selector_analytic_tolerance
            ).all(dim=-1) & selector_paired_valid[..., 4:].all(dim=-1)
            selector_decision_valid = (
                selector_changed & rank_valid[:, None] & selector_observable
            )
            selector_teacher_safe = (
                selector_decision_valid
                & selector_learned_safe
                & selector_analytic_safe
            )
            selector_decision_weight = torch.where(
                selector_teacher_safe,
                selector_decision_logit.new_full(
                    (), self.selector_positive_weight
                ),
                selector_decision_logit.new_full(
                    (), self.selector_false_takeover_weight
                ),
            )
            selector_decision_bce = F.binary_cross_entropy_with_logits(
                selector_decision_logit,
                selector_teacher_safe.to(base.dtype),
                reduction="none",
            )
            selector_decision_valid_float = selector_decision_valid.to(base.dtype)
            selector_decision_loss = (
                selector_decision_bce
                * selector_decision_weight
                * selector_decision_valid_float
            ).sum() / (
                selector_decision_weight * selector_decision_valid_float
            ).sum().clamp_min(1.0)
            losses["rcpc_loss_selector_decision"] = (
                selector_decision_loss * self.selector_decision_loss_weight
            )

            selector_utility_weight = self.sequential_selector.utility_weights.to(
                device=base.device, dtype=base.dtype
            ).view(1, 1, 4)
            selector_teacher_utility = (
                selector_metric_improvement_target[..., :4]
                * selector_utility_weight
            ).sum(dim=-1)
            selector_teacher_rank = selector_teacher_utility.masked_fill(
                ~selector_teacher_safe, -20.0
            )
            selector_teacher_rank = selector_teacher_rank.scatter(
                1,
                incumbent_selected[:, None],
                selector_teacher_rank.new_zeros(batch_size, 1),
            )
            selector_predicted_rank = output["selector_candidate_utility"][
                batch_index, command
            ]
            selector_predicted_rank = selector_predicted_rank.scatter(
                1,
                incumbent_selected[:, None],
                selector_predicted_rank.new_zeros(batch_size, 1),
            )
            selector_teacher_probability = torch.softmax(
                selector_teacher_rank / self.selector_listwise_temperature,
                dim=-1,
            )
            selector_listwise_per_sample = -(
                selector_teacher_probability
                * F.log_softmax(
                    selector_predicted_rank
                    / self.selector_listwise_temperature,
                    dim=-1,
                )
            ).sum(dim=-1)
            selector_sample_valid = (
                rank_valid
                & selector_observable.any(dim=-1)
            ).to(base.dtype)
            selector_listwise_loss = (
                selector_listwise_per_sample * selector_sample_valid
            ).sum() / selector_sample_valid.sum().clamp_min(1.0)
            losses["rcpc_loss_selector_listwise"] = (
                selector_listwise_loss * self.selector_listwise_loss_weight
            )
            selector_direct_logit = None
            if "selector_takeover_logit" in output:
                selector_direct_logit = output["selector_takeover_logit"][
                    batch_index, command
                ]
                direct_valid = selector_decision_valid.to(base.dtype)
                direct_weight = torch.where(
                    selector_teacher_safe,
                    selector_direct_logit.new_ones(()),
                    selector_direct_logit.new_full(
                        (), self.selector_direct_negative_weight
                    ),
                )
                direct_bce = F.binary_cross_entropy_with_logits(
                    selector_direct_logit,
                    selector_teacher_safe.to(base.dtype),
                    reduction="none",
                )
                direct_safety_loss = (
                    direct_bce * direct_weight * direct_valid
                ).sum() / (
                    direct_weight * direct_valid
                ).sum().clamp_min(1.0)
                losses["rcpc_loss_selector_direct_safety"] = (
                    direct_safety_loss
                    * self.selector_direct_safety_loss_weight
                )

                positive_pair = selector_teacher_safe[:, :, None]
                negative_pair = (
                    selector_decision_valid & (~selector_teacher_safe)
                )[:, None, :]
                pair_valid = positive_pair & negative_pair
                direct_pair_margin = (
                    selector_direct_logit[:, :, None]
                    - selector_direct_logit[:, None, :]
                )
                direct_rank_value = F.softplus(
                    self.selector_direct_rank_margin - direct_pair_margin
                )
                direct_rank_loss = (
                    direct_rank_value * pair_valid.to(base.dtype)
                ).sum() / pair_valid.to(base.dtype).sum().clamp_min(1.0)
                losses["rcpc_loss_selector_direct_rank"] = (
                    direct_rank_loss * self.selector_direct_rank_loss_weight
                )
            selector_margin_score = None
            selector_margin_prediction = None
            if "selector_margin_prediction" in output:
                selector_margin_prediction = output[
                    "selector_margin_prediction"
                ][batch_index, command]
                selector_margin_score = output[
                    "selector_margin_safety_score"
                ][batch_index, command]

                margin_regression_valid = (
                    selector_paired_valid[..., :4]
                    & selector_decision_valid[..., None]
                )
                margin_regression_valid_float = margin_regression_valid.to(
                    base.dtype
                )
                margin_regression_error = F.smooth_l1_loss(
                    selector_margin_prediction,
                    selector_metric_improvement_target[..., :4],
                    reduction="none",
                    beta=0.05,
                )
                margin_regression_loss = (
                    margin_regression_error * margin_regression_valid_float
                ).sum() / margin_regression_valid_float.sum().clamp_min(1.0)
                losses["rcpc_loss_selector_margin_regression"] = (
                    margin_regression_loss
                    * self.selector_margin_regression_loss_weight
                )

                margin_joint_weight = torch.where(
                    selector_teacher_safe,
                    selector_margin_score.new_full(
                        (), self.selector_margin_positive_weight
                    ),
                    selector_margin_score.new_ones(()),
                )
                margin_joint_bce = F.binary_cross_entropy_with_logits(
                    selector_margin_score,
                    selector_teacher_safe.to(base.dtype),
                    reduction="none",
                )
                margin_joint_valid = selector_decision_valid.to(base.dtype)
                margin_joint_loss = (
                    margin_joint_bce
                    * margin_joint_weight
                    * margin_joint_valid
                ).sum() / (
                    margin_joint_weight * margin_joint_valid
                ).sum().clamp_min(1.0)
                losses["rcpc_loss_selector_margin_joint"] = (
                    margin_joint_loss * self.selector_margin_joint_loss_weight
                )

                margin_safe_mask = selector_teacher_safe
                margin_unsafe_mask = (
                    selector_decision_valid & (~selector_teacher_safe)
                )
                finite_floor = selector_margin_score.new_full((), -1.0e4)
                best_safe_score = selector_margin_score.masked_fill(
                    ~margin_safe_mask, finite_floor
                ).max(dim=-1).values
                hardest_unsafe_score = selector_margin_score.masked_fill(
                    ~margin_unsafe_mask, finite_floor
                ).max(dim=-1).values
                margin_rank_valid = (
                    margin_safe_mask.any(dim=-1)
                    & margin_unsafe_mask.any(dim=-1)
                )
                margin_rank_value = F.softplus(
                    self.selector_margin_rank_margin
                    + hardest_unsafe_score
                    - best_safe_score
                )
                margin_rank_valid_float = margin_rank_valid.to(base.dtype)
                margin_rank_loss = (
                    margin_rank_value * margin_rank_valid_float
                ).sum() / margin_rank_valid_float.sum().clamp_min(1.0)
                losses["rcpc_loss_selector_margin_rank"] = (
                    margin_rank_loss * self.selector_margin_rank_loss_weight
                )

                margin_candidate_valid = selector_decision_valid.any(dim=-1)
                margin_no_safe = (
                    margin_candidate_valid & (~margin_safe_mask.any(dim=-1))
                )
                best_valid_score = selector_margin_score.masked_fill(
                    ~selector_decision_valid, finite_floor
                ).max(dim=-1).values
                margin_abstention_value = F.softplus(
                    best_valid_score - self.selector_margin_abstention_target
                )
                margin_no_safe_float = margin_no_safe.to(base.dtype)
                margin_abstention_loss = (
                    margin_abstention_value * margin_no_safe_float
                ).sum() / margin_no_safe_float.sum().clamp_min(1.0)
                losses["rcpc_loss_selector_margin_abstention"] = (
                    margin_abstention_loss
                    * self.selector_margin_abstention_loss_weight
                )
            selector_diagnostics = {
                "metric_improvement_target": selector_metric_improvement_target,
                "paired_metric_valid": selector_paired_valid,
                "lcb": selector_lcb,
                "teacher_safe": selector_teacher_safe,
                "changed": selector_changed,
                "decision_valid": selector_decision_valid,
                "decision_logit": selector_decision_logit,
                "direct_logit": selector_direct_logit,
                "margin_prediction": selector_margin_prediction,
                "margin_score": selector_margin_score,
                "incumbent": incumbent_selected,
            }

        if self.refiner_head is not None:
            raw_refined = output["raw_refined_cumulative"].reshape(
                batch_size,
                self.num_commands,
                self.num_modes,
                self.horizon_steps,
                2,
            )[batch_index, command]
            oracle_index = teacher_cost.argmin(dim=-1)
            oracle_refined = raw_refined[batch_index, oracle_index]
            gt_cumulative = targets["gt_cumulative"]
            gt_mask = targets["gt_mask"].to(base.dtype)
            horizon_weight = torch.linspace(
                0.5,
                1.5,
                self.horizon_steps,
                device=base.device,
                dtype=base.dtype,
            ).view(1, self.horizon_steps)
            refine_weight = gt_mask * horizon_weight
            refine_error = F.smooth_l1_loss(
                oracle_refined, gt_cumulative, reduction="none", beta=0.5
            ).sum(dim=-1)
            refine_loss = (refine_error * refine_weight).sum() / refine_weight.sum().clamp_min(1.0)
            residual = output["residual_cumulative"]
            residual_magnitude = residual.abs().mean()
            residual_smoothness = (
                residual[..., 1:, :] - residual[..., :-1, :]
            ).abs().mean()
            losses["rcpc_loss_refine"] = refine_loss * self.refine_loss_weight
            losses["rcpc_loss_residual"] = (
                residual_magnitude + residual_smoothness
            ) * self.residual_loss_weight

        with torch.no_grad():
            evaluated_score = output["calibrated_logits"].reshape(
                batch_size, self.num_commands, self.num_modes
            )[batch_index, command]
            selected = evaluated_score.argmax(dim=-1)
            oracle = teacher_cost.argmin(dim=-1)
            selected_cost = teacher_cost.gather(1, selected[:, None]).squeeze(1)
            oracle_cost = teacher_cost.gather(1, oracle[:, None]).squeeze(1)
            denom = valid_float.sum().clamp_min(1.0)
            losses.update(
                {
                    "rcpc_top1_oracle": (
                        (selected == oracle).to(base.dtype) * valid_float
                    ).sum()
                    / denom,
                    "rcpc_mean_regret": (
                        (selected_cost - oracle_cost) * valid_float
                    ).sum()
                    / denom,
                    "rcpc_score_spearman": (
                        self._score_spearman(evaluated_score, teacher_cost)
                        * valid_float
                    ).sum()
                    / denom,
                    "rcpc_collision_target_coverage": targets[
                        "collision_available"
                    ].to(base.dtype).mean(),
                    "rcpc_history_valid_ratio": output["history_valid"].to(
                        base.dtype
                    ).mean(),
                    "rcpc_calibration_available_ratio": output[
                        "calibration_available"
                    ].to(base.dtype).mean(),
                    "rcpc_score_delta_abs": output["score_delta"].abs().mean(),
                    "rcpc_proposed_score_delta_abs": output[
                        "proposed_score_delta"
                    ].abs().mean(),
                    "rcpc_analytic_gain_abs": self.analytic_gain.detach().abs().mean(),
                }
            )
            if gate_diagnostics is not None:
                gate_probability = torch.sigmoid(gate_diagnostics["logits"])
                gate_takeover = (
                    gate_probability >= self.trust_gate_threshold
                ) & gate_diagnostics["changed"] & rank_valid
                gate_positive = gate_diagnostics["target"] & rank_valid
                true_takeover = gate_takeover & gate_positive
                harmful_takeover = gate_takeover & (~gate_positive)
                losses.update(
                    {
                        "rcpc_trust_gate_target_rate": (
                            gate_positive.to(base.dtype).sum() / denom
                        ),
                        "rcpc_trust_gate_takeover_rate": (
                            gate_takeover.to(base.dtype).sum() / denom
                        ),
                        "rcpc_trust_gate_precision": (
                            true_takeover.to(base.dtype).sum()
                            / gate_takeover.to(base.dtype).sum().clamp_min(1.0)
                        ),
                        "rcpc_trust_gate_recall": (
                            true_takeover.to(base.dtype).sum()
                            / gate_positive.to(base.dtype).sum().clamp_min(1.0)
                        ),
                        "rcpc_trust_gate_harmful_rate": (
                            harmful_takeover.to(base.dtype).sum() / denom
                        ),
                        "rcpc_trust_gate_teacher_gain": (
                            (
                                gate_diagnostics["teacher_improvement"]
                                * gate_takeover.to(base.dtype)
                            ).sum()
                            / gate_takeover.to(base.dtype).sum().clamp_min(1.0)
                        ),
                    }
                )
            if risk_diagnostics is not None:
                risk_takeover = (
                    output["risk_controller_applied"][batch_index, command]
                    & rank_valid
                )
                risk_positive = risk_diagnostics["teacher_safe"] & rank_valid
                true_takeover = risk_takeover & risk_positive
                harmful_takeover = risk_takeover & (~risk_positive)
                paired_valid = risk_diagnostics["paired_metric_valid"][..., :4]
                coverage_denom = paired_valid.to(base.dtype).sum().clamp_min(1.0)
                lower_bound_covered = (
                    risk_diagnostics["metric_improvement_target"][..., :4]
                    >= risk_diagnostics["lcb"]
                ) & paired_valid
                takeover_denom = risk_takeover.to(base.dtype).sum().clamp_min(1.0)
                losses.update(
                    {
                        "rcpc_risk_target_rate": (
                            risk_positive.to(base.dtype).sum() / denom
                        ),
                        "rcpc_risk_predicted_safe_rate": (
                            output["risk_predicted_pareto_safe"][
                                batch_index, command
                            ].to(base.dtype).sum()
                            / denom
                        ),
                        "rcpc_risk_structural_safe_rate": (
                            output["risk_structural_safe"][
                                batch_index, command
                            ].to(base.dtype).sum()
                            / denom
                        ),
                        "rcpc_risk_changed_rate": (
                            risk_diagnostics["changed"].to(base.dtype).sum()
                            / denom
                        ),
                        "rcpc_risk_structural_changed_rate": (
                            (
                                output["risk_structural_safe"][
                                    batch_index, command
                                ]
                                & risk_diagnostics["changed"]
                                & rank_valid
                            ).to(base.dtype).sum()
                            / denom
                        ),
                        "rcpc_risk_takeover_rate": (
                            risk_takeover.to(base.dtype).sum() / denom
                        ),
                        "rcpc_risk_precision": (
                            true_takeover.to(base.dtype).sum() / takeover_denom
                        ),
                        "rcpc_risk_recall": (
                            true_takeover.to(base.dtype).sum()
                            / risk_positive.to(base.dtype).sum().clamp_min(1.0)
                        ),
                        "rcpc_risk_harmful_rate": (
                            harmful_takeover.to(base.dtype).sum() / denom
                        ),
                        "rcpc_risk_lcb_empirical_coverage": (
                            lower_bound_covered.to(base.dtype).sum()
                            / coverage_denom
                        ),
                        "rcpc_risk_takeover_l2_6s_gain": (
                            risk_diagnostics["metric_improvement_target"][..., 2]
                            * risk_takeover.to(base.dtype)
                        ).sum()
                        / takeover_denom,
                        "rcpc_risk_mean_displacement": output[
                            "risk_candidate_displacement"
                        ][batch_index, command].mean(),
                        "rcpc_risk_lcb_4s_mean": risk_diagnostics["lcb"][
                            ..., 0
                        ].mean(),
                        "rcpc_risk_lcb_5s_mean": risk_diagnostics["lcb"][
                            ..., 1
                        ].mean(),
                        "rcpc_risk_lcb_6s_mean": risk_diagnostics["lcb"][
                            ..., 2
                        ].mean(),
                        "rcpc_risk_lcb_collision_mean": risk_diagnostics[
                            "lcb"
                        ][..., 3].mean(),
                        "rcpc_risk_lcb_min_mean": risk_diagnostics["lcb"].min(
                            dim=-1
                        ).values.mean(),
                    }
                )
            if selector_diagnostics is not None:
                selector_takeover = (
                    output["selector_applied"][batch_index, command]
                    & rank_valid
                )
                selector_winner = output["selector_winner"][
                    batch_index, command
                ]
                selector_selected_teacher_safe = torch.gather(
                    selector_diagnostics["teacher_safe"],
                    1,
                    selector_winner[:, None],
                ).squeeze(1)
                selector_true_takeover = (
                    selector_takeover & selector_selected_teacher_safe
                )
                selector_harmful_takeover = (
                    selector_takeover & (~selector_selected_teacher_safe)
                )
                selector_positive_sample = selector_diagnostics[
                    "teacher_safe"
                ].any(dim=-1) & rank_valid
                selector_paired_valid = selector_diagnostics[
                    "paired_metric_valid"
                ][..., :4]
                selector_coverage_denom = selector_paired_valid.to(
                    base.dtype
                ).sum().clamp_min(1.0)
                selector_lower_bound_covered = (
                    selector_diagnostics["metric_improvement_target"][..., :4]
                    >= selector_diagnostics["lcb"]
                ) & selector_paired_valid
                selector_takeover_denom = selector_takeover.to(
                    base.dtype
                ).sum().clamp_min(1.0)
                selector_selected_l2_gain = torch.gather(
                    selector_diagnostics["metric_improvement_target"][..., 2],
                    1,
                    selector_winner[:, None],
                ).squeeze(1)
                selector_displacement = output[
                    "selector_candidate_displacement"
                ][batch_index, command]
                selector_selected_displacement = torch.gather(
                    selector_displacement,
                    1,
                    selector_winner[:, None, None].expand(
                        batch_size, 1, self.horizon_steps
                    ),
                ).squeeze(1)
                selector_candidate_denom = selector_diagnostics[
                    "decision_valid"
                ].to(base.dtype).sum().clamp_min(1.0)
                losses.update(
                    {
                        "rcpc_selector_target_rate": (
                            selector_positive_sample.to(base.dtype).sum() / denom
                        ),
                        "rcpc_selector_predicted_safe_candidate_rate": (
                            (
                                output["selector_predicted_pareto_safe"][
                                    batch_index, command
                                ]
                                & selector_diagnostics["decision_valid"]
                            ).to(base.dtype).sum()
                            / selector_candidate_denom
                        ),
                        "rcpc_selector_structural_safe_candidate_rate": (
                            (
                                output["selector_structural_safe"][
                                    batch_index, command
                                ]
                                & selector_diagnostics["decision_valid"]
                            ).to(base.dtype).sum()
                            / selector_candidate_denom
                        ),
                        "rcpc_selector_takeover_rate": (
                            selector_takeover.to(base.dtype).sum() / denom
                        ),
                        "rcpc_selector_precision": (
                            selector_true_takeover.to(base.dtype).sum()
                            / selector_takeover_denom
                        ),
                        "rcpc_selector_recall": (
                            selector_true_takeover.to(base.dtype).sum()
                            / selector_positive_sample.to(base.dtype).sum().clamp_min(1.0)
                        ),
                        "rcpc_selector_harmful_rate": (
                            selector_harmful_takeover.to(base.dtype).sum() / denom
                        ),
                        "rcpc_selector_lcb_empirical_coverage": (
                            selector_lower_bound_covered.to(base.dtype).sum()
                            / selector_coverage_denom
                        ),
                        "rcpc_selector_takeover_l2_6s_gain": (
                            selector_selected_l2_gain
                            * selector_takeover.to(base.dtype)
                        ).sum()
                        / selector_takeover_denom,
                        "rcpc_selector_mean_displacement": (
                            selector_selected_displacement.mean(dim=-1)
                            * selector_takeover.to(base.dtype)
                        ).sum()
                        / selector_takeover_denom,
                        "rcpc_selector_endpoint_displacement": (
                            selector_selected_displacement[..., -1]
                            * selector_takeover.to(base.dtype)
                        ).sum()
                        / selector_takeover_denom,
                        "rcpc_selector_innovation_gate_mean": output[
                            "selector_innovation_gate"
                        ][batch_index, command].mean(),
                        "rcpc_selector_lcb_4s_mean": selector_diagnostics[
                            "lcb"
                        ][..., 0].mean(),
                        "rcpc_selector_lcb_5s_mean": selector_diagnostics[
                            "lcb"
                        ][..., 1].mean(),
                        "rcpc_selector_lcb_6s_mean": selector_diagnostics[
                            "lcb"
                        ][..., 2].mean(),
                        "rcpc_selector_lcb_collision_mean": selector_diagnostics[
                            "lcb"
                        ][..., 3].mean(),
                    }
                )
                if selector_diagnostics["direct_logit"] is not None:
                    direct_probability = torch.sigmoid(
                        selector_diagnostics["direct_logit"]
                    )
                    proposed_winner = output["selector_direct_proposed_winner"][
                        batch_index, command
                    ]
                    proposed_safe = torch.gather(
                        selector_diagnostics["teacher_safe"],
                        1,
                        proposed_winner[:, None],
                    ).squeeze(1)
                    direct_eligible_any = output[
                        "selector_direct_eligible"
                    ][batch_index, command].any(dim=-1) & rank_valid
                    losses.update(
                        {
                            "rcpc_selector_direct_probability_mean": (
                                direct_probability
                                * selector_diagnostics["decision_valid"].to(
                                    base.dtype
                                )
                            ).sum()
                            / selector_candidate_denom,
                            "rcpc_selector_direct_top1_precision": (
                                proposed_safe
                                & direct_eligible_any
                            ).to(base.dtype).sum()
                            / direct_eligible_any.to(base.dtype).sum().clamp_min(1.0),
                            "rcpc_selector_direct_candidate_positive_rate": (
                                selector_diagnostics["teacher_safe"]
                            ).to(base.dtype).sum()
                            / selector_candidate_denom,
                        }
                    )
                if selector_diagnostics["margin_score"] is not None:
                    margin_probability = torch.sigmoid(
                        selector_diagnostics["margin_score"]
                    )
                    margin_proposed_winner = output[
                        "selector_proposed_winner"
                    ][batch_index, command]
                    margin_proposed_safe = torch.gather(
                        selector_diagnostics["teacher_safe"],
                        1,
                        margin_proposed_winner[:, None],
                    ).squeeze(1)
                    margin_eligible_any = output[
                        "selector_direct_eligible"
                    ][batch_index, command].any(dim=-1) & rank_valid
                    margin_regression_valid = (
                        selector_diagnostics["paired_metric_valid"][..., :4]
                        & selector_diagnostics["decision_valid"][..., None]
                    )
                    margin_regression_error = (
                        selector_diagnostics["margin_prediction"]
                        - selector_diagnostics["metric_improvement_target"][
                            ..., :4
                        ]
                    ).abs()
                    losses.update(
                        {
                            "rcpc_selector_margin_probability_mean": (
                                margin_probability
                                * selector_diagnostics["decision_valid"].to(
                                    base.dtype
                                )
                            ).sum()
                            / selector_candidate_denom,
                            "rcpc_selector_margin_top1_precision": (
                                margin_proposed_safe & margin_eligible_any
                            ).to(base.dtype).sum()
                            / margin_eligible_any.to(base.dtype).sum().clamp_min(1.0),
                            "rcpc_selector_margin_candidate_positive_rate": (
                                selector_diagnostics["teacher_safe"]
                            ).to(base.dtype).sum()
                            / selector_candidate_denom,
                            "rcpc_selector_margin_regression_mae": (
                                margin_regression_error
                                * margin_regression_valid.to(base.dtype)
                            ).sum()
                            / margin_regression_valid.to(base.dtype).sum().clamp_min(1.0),
                        }
                    )

        for name, value in losses.items():
            if not torch.isfinite(value).all():
                raise FloatingPointError("RCPC loss %s contains NaN/Inf" % name)
        return losses


class RCPCPlanningAdapter(RetrospectivelyCalibratedParetoCritic):
    """Stable public adapter name used by the MomAD planning head."""

    pass
