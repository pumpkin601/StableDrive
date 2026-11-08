from __future__ import annotations

import torch
import torch.nn as nn

from .causal_reliability_gate import RetrospectiveReliabilityGate
from .selective_momentum_memory import RealMamba, SelectiveMomentumMemory


class CausalSMMAdapter(nn.Module):
    """Strict causal wrapper around the A1 SMM module.

    The adapter intentionally owns no history cache. Its inputs must already be
    audited rank-local causal model predictions supplied by the C0 head.
    """

    def __init__(
        self,
        embed_dims=256,
        num_modes=18,
        history_len=1,
        use_real_mamba=True,
        require_real_mamba=True,
        allow_mamba_fallback=False,
        alpha_init=1e-3,
        d_state=16,
        d_conv=4,
        expand=2,
        use_reliability_gate=False,
        beta_policy="none",
        allocation_policy=None,
        reliability_gate_cfg=None,
        cr_v1_2_enabled=False,
        cr_v1_2_fixed_one=False,
    ):
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.num_modes = int(num_modes)
        self.history_len = int(history_len)
        self.use_real_mamba = bool(use_real_mamba)
        self.require_real_mamba = bool(require_real_mamba)
        self.allow_mamba_fallback = bool(allow_mamba_fallback)
        self.cr_v1_2_enabled = bool(cr_v1_2_enabled)
        self.cr_v1_2_fixed_one = bool(cr_v1_2_fixed_one)
        self.use_reliability_gate = bool(use_reliability_gate) and not self.cr_v1_2_fixed_one
        self.beta_policy = str(beta_policy)
        self.allocation_policy = (
            str(allocation_policy)
            if allocation_policy is not None
            else self.beta_policy
        )
        if self.cr_v1_2_fixed_one:
            self.allocation_policy = "fixed_one_bypass"
            self.beta_policy = "fixed_one_bypass"
        self.forward_call_count = 0
        self.real_mamba_forward_call_count = 0
        self.fallback_activated = False
        self._audit_records = []
        self._reliability_audit_records = []

        if self.history_len < 1:
            raise ValueError("Causal SMM requires history_len >= 1")
        if self.require_real_mamba and not self.use_real_mamba:
            raise ValueError("require_real_mamba=True requires use_real_mamba=True")
        if self.require_real_mamba and self.allow_mamba_fallback:
            raise ValueError("require_real_mamba=True forbids fallback")
        if self.require_real_mamba and RealMamba is None:
            raise ImportError("mamba_ssm RealMamba import failed; fallback is forbidden")

        self.memory = SelectiveMomentumMemory(
            embed_dims=self.embed_dims,
            num_modes=self.num_modes,
            history_len=self.history_len,
            use_real_mamba=self.use_real_mamba,
            fallback_if_unavailable=self.allow_mamba_fallback,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        if self.require_real_mamba and not bool(self.memory.using_real_mamba):
            raise RuntimeError("C1 requires Real Mamba, but SMM backend is fallback")
        self.fallback_activated = not bool(self.memory.using_real_mamba)
        with torch.no_grad():
            self.memory.alpha.fill_(float(alpha_init))
        self.reliability_gate = None
        if self.use_reliability_gate:
            if self.allocation_policy != "learned" and self.beta_policy != "learned":
                raise ValueError(
                    "use_reliability_gate=True requires allocation_policy='learned'"
                )
            cfg = dict(reliability_gate_cfg or {})
            cfg.setdefault("embed_dims", self.embed_dims)
            if self.cr_v1_2_enabled:
                cfg["cr_v1_2_enabled"] = True
            self.reliability_gate = RetrospectiveReliabilityGate(**cfg)
        elif self.allocation_policy not in ("none", "fixed_one_bypass"):
            raise ValueError(
                "allocation_policy must be one of {'none', 'fixed_one_bypass', 'learned'}"
            )

    @property
    def real_mamba_imported(self):
        return RealMamba is not None

    @property
    def real_mamba_instantiated(self):
        return bool(getattr(self.memory, "using_real_mamba", False))

    @property
    def alpha(self):
        return self.memory.alpha

    def _zero_reliability_dependency(self, reference):
        if self.reliability_gate is None:
            return reference.new_zeros(())
        zero = reference.new_zeros(())
        for param in self.reliability_gate.parameters():
            zero = zero + param.to(dtype=reference.dtype).sum() * 0.0
        return zero

    def forward(
        self,
        current_query,
        history_query,
        history_score,
        history_valid_mask,
        reliability_features=None,
        reliability_availability=None,
    ):
        if self.require_real_mamba and not self.real_mamba_instantiated:
            raise RuntimeError("Real Mamba is required but not active")
        if self.fallback_activated:
            raise RuntimeError("C1 fallback path activated unexpectedly")
        self.forward_call_count += 1
        valid_history = bool(history_valid_mask.detach().to(dtype=torch.bool).any().item())
        history_allocation_weight = None
        allocation_stats = {
            "allocation_policy": self.allocation_policy,
            "cr_v1_2_enabled": bool(self.cr_v1_2_enabled),
            "cr_v1_2_fixed_one": bool(self.cr_v1_2_fixed_one),
            "fixed_one_bypass": bool(
                self.allocation_policy in ("none", "fixed_one_bypass")
            ),
            "reliability_gate_forward_called": False,
            "release_forward_count": 0,
            "penalty_applied_count": 0,
            "allocation_forward_count": 0,
        }
        if self.allocation_policy == "learned" or self.beta_policy == "learned":
            if self.reliability_gate is None:
                raise RuntimeError("allocation_policy='learned' requires reliability_gate")
            if reliability_features is None or reliability_availability is None:
                raise RuntimeError(
                    "causal allocation requires reliability_features and "
                    "reliability_availability"
                )
            history_allocation_weight, allocation_stats = self.reliability_gate(
                current_query=current_query,
                history_query=history_query,
                reliability_features=reliability_features,
                reliability_availability=reliability_availability,
                history_valid_mask=history_valid_mask,
            )
            allocation_stats["allocation_policy"] = (
                "cr_v1_2_evidence_gated_recency_safe"
                if self.cr_v1_2_enabled
                else self.allocation_policy
            )
            allocation_stats["cr_v1_2_enabled"] = bool(self.cr_v1_2_enabled)
            allocation_stats["cr_v1_2_fixed_one"] = False
            allocation_stats["fixed_one_bypass"] = False
            allocation_stats["reliability_gate_forward_called"] = True
            allocation_stats["allocation_forward_count"] = 1
            if self.cr_v1_2_enabled:
                cr_v1_2_audit = allocation_stats.get("cr_v1_2_allocation_audit", {})
                allocation_stats["release_forward_count"] = int(
                    cr_v1_2_audit.get("release_forward_count", 0)
                )
                allocation_stats["penalty_applied_count"] = int(
                    any(
                        (rec.get("effective_stale_penalty_i") or 0.0) > 0.0
                        for rec in cr_v1_2_audit.get("slot_records", [])
                    )
                )
            else:
                allocation_stats["release_forward_count"] = 0
                allocation_stats["penalty_applied_count"] = 0
        allocation_audit = allocation_stats.get("cr_v1_2_allocation_audit")
        memory_kwargs = dict(
            current_query=current_query,
            history_query=history_query,
            history_score=history_score,
            history_valid_mask=history_valid_mask,
        )
        if self.cr_v1_2_enabled and history_allocation_weight is not None:
            memory_kwargs["history_weight"] = history_allocation_weight
            memory_kwargs["allocation_audit"] = allocation_audit
        else:
            memory_kwargs["history_beta"] = history_allocation_weight
        output = self.memory(
            **memory_kwargs,
        )
        zero_dependency_activated = bool(
            (self.allocation_policy == "learned" or self.beta_policy == "learned")
            and not valid_history
        )
        if zero_dependency_activated:
            output = output + self._zero_reliability_dependency(output)
        if valid_history:
            self.real_mamba_forward_call_count += 1
        record = {
            "SMM_enabled": True,
            "RealMamba_imported": bool(self.real_mamba_imported),
            "RealMamba_instantiated": bool(self.real_mamba_instantiated),
            "RealMamba_forward_called": bool(valid_history),
            "fallback_activated": bool(self.fallback_activated),
            "SMM_input_shape": list(current_query.shape),
            "SMM_output_shape": list(output.shape),
            "SMM_input_dtype": str(current_query.dtype),
            "SMM_output_dtype": str(output.dtype),
            "SMM_output_finite": bool(torch.isfinite(output).all().item()),
            "SMM_valid_history_count": int(
                history_valid_mask.detach().to(dtype=torch.bool).sum().item()
            ),
            "SMM_history_len": int(self.history_len),
            "SMM_history_query_shape": list(history_query.shape)
            if history_query is not None
            else None,
            "SMM_history_valid_mask_shape": list(history_valid_mask.shape)
            if history_valid_mask is not None
            else None,
            "SMM_group_records": list(
                getattr(self.memory, "last_group_records", [])
            ),
            "SMM_alpha": float(self.alpha.detach().float().cpu().item()),
            "CR_reliability": allocation_stats,
            "CR_v1_2_allocation": allocation_stats
            if self.cr_v1_2_enabled
            else None,
            "CR_valid_history_batch": bool(valid_history),
            "CR_zero_reliability_dependency_activated": zero_dependency_activated,
        }
        allocation_stats["valid_history_batch"] = bool(valid_history)
        allocation_stats["zero_reliability_dependency_activated"] = zero_dependency_activated
        self._audit_records.append(record)
        self._reliability_audit_records.append(allocation_stats)
        if len(self._audit_records) > 4096:
            self._audit_records = self._audit_records[-4096:]
        if len(self._reliability_audit_records) > 4096:
            self._reliability_audit_records = self._reliability_audit_records[-4096:]
        return output

    def consume_audit(self):
        records = self._audit_records
        reliability_records = self._reliability_audit_records
        self._audit_records = []
        self._reliability_audit_records = []
        return {
            "records": records,
            "reliability_records": reliability_records,
            "RealMamba_imported": bool(self.real_mamba_imported),
            "RealMamba_instantiated": bool(self.real_mamba_instantiated),
            "RealMamba_forward_call_count": int(self.real_mamba_forward_call_count),
            "SMM_forward_call_count": int(self.forward_call_count),
            "fallback_activated": bool(self.fallback_activated),
            "beta_policy": self.beta_policy,
            "allocation_policy": self.allocation_policy,
            "cr_v1_2_enabled": bool(self.cr_v1_2_enabled),
            "cr_v1_2_fixed_one": bool(self.cr_v1_2_fixed_one),
            "use_reliability_gate": bool(self.use_reliability_gate),
            "num_parameters": int(sum(p.numel() for p in self.parameters())),
            "num_trainable_parameters": int(
                sum(p.numel() for p in self.parameters() if p.requires_grad)
            ),
        }
