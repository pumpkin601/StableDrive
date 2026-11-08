from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path


SCHEMA_VERSION = "smm_observability_v1"

ACTION_INVOKED = "SMM_INVOKED"
ACTION_SKIPPED = "SMM_SKIPPED"

REASON_INVOKED = "H1_HISTORY_VALID_AND_SMM_INVOKED"
REASON_COLD_START = "COLD_START_NO_PRIOR_H1"
REASON_SCENE_BOUNDARY = "SCENE_BOUNDARY_RESET"
REASON_TIMESTAMP_DISCONTINUITY = "TIMESTAMP_OR_ORDER_DISCONTINUITY"
REASON_H1_VALID_FALSE = "H1_VALID_MASK_FALSE"
REASON_CACHE_EVICTION = "CACHE_CLEAR_OR_EVICTION"
REASON_EXPLICIT_BYPASS = "EXPLICIT_SMM_BYPASS_BY_DESIGN"
REASON_OTHER = "OTHER_EVIDENCED_REASON"

APPROVED_REASON_CODES = (
    REASON_INVOKED,
    REASON_COLD_START,
    REASON_SCENE_BOUNDARY,
    REASON_TIMESTAMP_DISCONTINUITY,
    REASON_H1_VALID_FALSE,
    REASON_CACHE_EVICTION,
    REASON_EXPLICIT_BYPASS,
    REASON_OTHER,
)

FAIL_CLOSED_FORMAL_REASONS = (
    REASON_EXPLICIT_BYPASS,
    REASON_OTHER,
)

REASON_PREDICATE_MAP = {
    REASON_INVOKED: "causal_smm_adapter.real_mamba_forward_count_incremented",
    REASON_COLD_START: "motion_planning_head._causal_miss_reason.scene_start_or_empty_cache",
    REASON_SCENE_BOUNDARY: "motion_planning_head._causal_miss_reason.scene_mismatch",
    REASON_TIMESTAMP_DISCONTINUITY: "motion_planning_head._causal_miss_reason.token_discontinuity",
    REASON_H1_VALID_FALSE: "motion_planning_head.causal_history_valid_false_after_lookup",
    REASON_CACHE_EVICTION: "motion_planning_head.causal_cache_management.clear_or_eviction",
    REASON_EXPLICIT_BYPASS: "formal_config.explicit_smm_bypass_by_design",
    REASON_OTHER: "manual_preapproved_other_evidenced_reason",
}

REQUIRED_EVENT_FIELDS = (
    "schema_version",
    "run_id",
    "rank",
    "world_size",
    "forward_event_id",
    "local_attempt_index",
    "local_successful_update_index_or_pending",
    "global_update_context",
    "action",
    "sample_token",
    "scene_token",
    "timestamp",
    "h1_cache_len_before",
    "h1_cache_len_after",
    "cached_scene_token_before",
    "cached_timestamp_before",
    "current_scene_matches_cached_scene",
    "timestamp_delta",
    "h1_valid_mask",
    "history_available",
    "cache_reset_flag",
    "cache_eviction_flag",
    "smm_invocation_count_before",
    "smm_invocation_count_after",
    "reason_code",
    "reason_predicate_id",
    "reason_evidence",
    "config_sha256",
    "source_hash_bundle_id",
)


class SMMObservabilityError(RuntimeError):
    pass


class SMMObservabilityObserver:
    """Append-only observer for rank-local SMM decision telemetry.

    This helper is intentionally stdlib-only and contains no tensor operations.
    It records evidence after the production H1/SMM branch has already made its
    decision. The only control-flow effect is fail-closed audit termination.
    """

    def __init__(
        self,
        enabled=False,
        fail_closed=True,
        output_subdir="audit",
        schema_version=SCHEMA_VERSION,
        flush_policy="line",
        run_id=None,
        config_sha256=None,
        source_hash_bundle_id=None,
        allow_other_evidenced_reason=False,
        allow_explicit_bypass=False,
    ):
        self.enabled = bool(enabled)
        self.fail_closed = bool(fail_closed)
        self.output_subdir = str(output_subdir)
        self.schema_version = str(schema_version)
        self.flush_policy = str(flush_policy)
        self.run_id = run_id or os.environ.get("C3A_FORMAL_RUN_NAME") or "UNSET_RUN_ID"
        self.config_sha256 = config_sha256 or os.environ.get(
            "C3A_FORMAL_EXPECTED_CONFIG_SHA", "UNSET_CONFIG_SHA256"
        )
        self.source_hash_bundle_id = source_hash_bundle_id or os.environ.get(
            "C3A_SOURCE_HASH_BUNDLE_ID", "UNSET_SOURCE_HASH_BUNDLE_ID"
        )
        self.allow_other_evidenced_reason = bool(allow_other_evidenced_reason)
        self.allow_explicit_bypass = bool(allow_explicit_bypass)
        self.local_attempt_index = 0
        self.event_count = 0
        self.invoked_count = 0
        self.skipped_count = 0

    def _rank_file(self, rank, name):
        run_dir = os.environ.get("SMM_OBSERVABILITY_RUN_DIR")
        if not run_dir:
            self._raise("SMM_OBSERVABILITY_RUN_DIR is required when SMM observability is enabled")
        out_dir = Path(run_dir) / self.output_subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        return out_dir / f"{name}_rank{int(rank)}.jsonl"

    def _raise(self, message):
        if self.fail_closed:
            raise SMMObservabilityError(f"SMM_OBSERVABILITY_FAIL_CLOSED: {message}")

    def _append_jsonl(self, path, payload):
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(payload, sort_keys=True) + "\n")
                if self.flush_policy == "line":
                    f.flush()
                    os.fsync(f.fileno())
        except Exception as exc:
            self._raise(f"telemetry write failure: {exc!r}")

    def _validate_event_schema(self, event):
        missing = [key for key in REQUIRED_EVENT_FIELDS if key not in event]
        if missing:
            self._raise(f"telemetry schema validation failure: missing {missing}")
            return False
        if event["action"] not in (ACTION_INVOKED, ACTION_SKIPPED):
            self._raise(f"invalid SMM action {event['action']!r}")
            return False
        if event["reason_code"] not in APPROVED_REASON_CODES:
            self._raise(f"reason_code absent from approved taxonomy: {event['reason_code']!r}")
            return False
        if not event["reason_predicate_id"]:
            self._raise("reason_predicate_id absent")
            return False
        if event["reason_code"] == REASON_OTHER and not self.allow_other_evidenced_reason:
            self._raise("SMM skip appears with OTHER_EVIDENCED_REASON without approval")
            return False
        if event["reason_code"] == REASON_EXPLICIT_BYPASS and not self.allow_explicit_bypass:
            self._raise("SMM skip appears with EXPLICIT_SMM_BYPASS_BY_DESIGN without approval")
            return False
        return True

    def _validate_reason_evidence(self, event):
        evidence = event.get("reason_evidence") or {}
        reason = event["reason_code"]
        if reason == REASON_INVOKED:
            ok = (
                event["action"] == ACTION_INVOKED
                and event["history_available"] is True
                and event["smm_invocation_count_after"] > event["smm_invocation_count_before"]
            )
        elif reason == REASON_COLD_START:
            ok = (
                event["action"] == ACTION_SKIPPED
                and int(event["h1_cache_len_before"]) == 0
                and event["history_available"] is False
            )
        elif reason == REASON_SCENE_BOUNDARY:
            ok = (
                event["action"] == ACTION_SKIPPED
                and event["cache_reset_flag"] is True
                and event["current_scene_matches_cached_scene"] is False
                and bool(evidence.get("scene_transition_evidence_present"))
            )
        elif reason == REASON_TIMESTAMP_DISCONTINUITY:
            ok = (
                event["action"] == ACTION_SKIPPED
                and bool(evidence.get("timestamp_or_order_predicate_failed"))
            )
        elif reason == REASON_H1_VALID_FALSE:
            ok = (
                event["action"] == ACTION_SKIPPED
                and bool(evidence.get("prior_history_exists"))
                and event["h1_valid_mask"] is False
            )
        elif reason == REASON_CACHE_EVICTION:
            ok = (
                event["action"] == ACTION_SKIPPED
                and event["cache_eviction_flag"] is True
                and bool(evidence.get("cache_management_evidence_present"))
            )
        else:
            ok = False
        if not ok:
            self._raise(f"reason evidence inconsistent for {reason}")
        return ok

    def _skip_reason_from_evidence(self, batch_records, h1_cache_len_before):
        miss_reasons = [rec.get("history_cache_miss_reason") for rec in batch_records]
        if int(h1_cache_len_before) == 0 and not any(
            bool(rec.get("history_cache_hit")) for rec in batch_records
        ):
            return REASON_COLD_START, {
                "predicate": "empty_cache_and_no_history_hit",
                "miss_reasons": miss_reasons,
            }
        if all(rec.get("is_scene_start") for rec in batch_records):
            return REASON_SCENE_BOUNDARY, {
                "scene_transition_evidence_present": True,
                "miss_reasons": miss_reasons,
            }
        if any(rec.get("cache_cross_scene_detected") for rec in batch_records):
            return REASON_SCENE_BOUNDARY, {
                "scene_transition_evidence_present": True,
                "miss_reasons": miss_reasons,
            }
        if any(
            reason in (
                "token_discontinuity",
                "prev_token_not_in_local_rank_cache",
                "rank_mismatch",
            )
            for reason in miss_reasons
        ):
            return REASON_TIMESTAMP_DISCONTINUITY, {
                "timestamp_or_order_predicate_failed": True,
                "miss_reasons": miss_reasons,
            }
        if any(rec.get("history_source_token") for rec in batch_records):
            return REASON_H1_VALID_FALSE, {
                "prior_history_exists": True,
                "miss_reasons": miss_reasons,
            }
        return REASON_OTHER, {
            "manual_preapproval_required": True,
            "miss_reasons": miss_reasons,
        }

    def record_decision(
        self,
        *,
        rank,
        world_size,
        h1_cache_len_before,
        h1_cache_len_after,
        batch_records,
        smm_invocation_count_before,
        smm_invocation_count_after,
        global_update_context="PENDING",
    ):
        if not self.enabled:
            return None
        self.local_attempt_index += 1
        event_id = str(uuid.uuid4())
        invoked = int(smm_invocation_count_after) > int(smm_invocation_count_before)
        action = ACTION_INVOKED if invoked else ACTION_SKIPPED
        history_available = any(bool(rec.get("history_cache_hit")) for rec in batch_records)
        h1_valid = bool(history_available)
        cached_scene_tokens = [
            rec.get("history_source_scene_token") for rec in batch_records
            if rec.get("history_source_scene_token") is not None
        ]
        cached_timestamps = [
            rec.get("history_source_timestamp") for rec in batch_records
            if rec.get("history_source_timestamp") is not None
        ]
        current_scene_matches_cached = all(
            not rec.get("history_source_scene_token")
            or rec.get("history_source_scene_token") == rec.get("scene_token")
            for rec in batch_records
        )
        if any(rec.get("is_scene_start") for rec in batch_records) and not history_available:
            current_scene_matches_cached = False
        timestamp_deltas = [
            rec.get("timestamp_delta") for rec in batch_records
            if rec.get("timestamp_delta") is not None
        ]
        if invoked:
            reason_code = REASON_INVOKED
            reason_evidence = {
                "valid_history_batch": True,
                "smm_invocation_counter_incremented": True,
            }
        else:
            reason_code, reason_evidence = self._skip_reason_from_evidence(
                batch_records,
                h1_cache_len_before,
            )
        event = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "rank": int(rank),
            "world_size": int(world_size),
            "forward_event_id": event_id,
            "local_attempt_index": int(self.local_attempt_index),
            "local_successful_update_index_or_pending": "PENDING",
            "global_update_context": global_update_context,
            "action": action,
            "sample_token": [rec.get("sample_token") for rec in batch_records],
            "scene_token": [rec.get("scene_token") for rec in batch_records],
            "timestamp": [rec.get("timestamp") for rec in batch_records],
            "h1_cache_len_before": int(h1_cache_len_before),
            "h1_cache_len_after": int(h1_cache_len_after),
            "cached_scene_token_before": cached_scene_tokens,
            "cached_timestamp_before": cached_timestamps,
            "current_scene_matches_cached_scene": bool(current_scene_matches_cached),
            "timestamp_delta": timestamp_deltas,
            "h1_valid_mask": bool(h1_valid),
            "history_available": bool(history_available),
            "cache_reset_flag": bool(any(rec.get("is_scene_start") for rec in batch_records)),
            "cache_eviction_flag": bool(any(rec.get("cache_eviction_flag") for rec in batch_records)),
            "smm_invocation_count_before": int(smm_invocation_count_before),
            "smm_invocation_count_after": int(smm_invocation_count_after),
            "reason_code": reason_code,
            "reason_predicate_id": REASON_PREDICATE_MAP.get(reason_code),
            "reason_evidence": reason_evidence,
            "config_sha256": self.config_sha256,
            "source_hash_bundle_id": self.source_hash_bundle_id,
            "created_unix_time": time.time(),
        }
        if self._validate_event_schema(event):
            self._validate_reason_evidence(event)
        path = self._rank_file(rank, "smm_events")
        self._append_jsonl(path, event)
        self.event_count += 1
        if invoked:
            self.invoked_count += 1
        else:
            self.skipped_count += 1
        return event

    def record_update_commit(
        self,
        *,
        rank,
        world_size,
        forward_event_id,
        local_successful_update_index,
        update_committed,
        skipped_or_failed_reason=None,
    ):
        if not self.enabled:
            return None
        payload = {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "rank": int(rank),
            "world_size": int(world_size),
            "forward_event_id": str(forward_event_id),
            "local_successful_update_index": int(local_successful_update_index)
            if update_committed
            else None,
            "update_committed": bool(update_committed),
            "skipped_or_failed_reason": skipped_or_failed_reason,
            "created_unix_time": time.time(),
        }
        self._append_jsonl(self._rank_file(rank, "smm_update_commit"), payload)
        return payload

    def summary(self):
        return {
            "schema_version": self.schema_version,
            "enabled": bool(self.enabled),
            "fail_closed": bool(self.fail_closed),
            "event_count": int(self.event_count),
            "invoked_count": int(self.invoked_count),
            "skipped_count": int(self.skipped_count),
            "approved_reason_codes": list(APPROVED_REASON_CODES),
            "reason_predicate_map": dict(REASON_PREDICATE_MAP),
        }
