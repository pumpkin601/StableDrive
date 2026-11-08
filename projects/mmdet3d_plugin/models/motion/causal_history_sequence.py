from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch


NEURAL_TENSOR_FIELDS = (
    "plan_query",
    "candidate_trajectory",
    "selected_planning",
    "candidate_logits",
)


@dataclass
class CausalHistoryEntry:
    sample_token: str
    prev_token: str
    scene_token: str
    rank_id: int
    timestamp: float
    cache_write_order: int
    history_source: str = "model_prediction"
    plan_query: Optional[torch.Tensor] = None
    candidate_trajectory: Optional[torch.Tensor] = None
    selected_planning: Optional[torch.Tensor] = None
    candidate_logits: Optional[torch.Tensor] = None
    ego_translation: Optional[Tuple[float, float, float]] = None
    ego_yaw: Optional[float] = None
    ego_velocity: Optional[Tuple[float, float]] = None
    ego_status: Optional[torch.Tensor] = None
    metadata: Dict[str, object] = field(default_factory=dict)

    def key(self) -> Tuple[str, str]:
        return self.scene_token, self.sample_token

    def tensor_fields(self) -> Dict[str, Optional[torch.Tensor]]:
        return {name: getattr(self, name) for name in NEURAL_TENSOR_FIELDS}

    def neural_tensors_detached(self) -> bool:
        for tensor in self.tensor_fields().values():
            if tensor is not None and tensor.requires_grad:
                return False
        if self.ego_status is not None and self.ego_status.requires_grad:
            return False
        return True

    def shape_dtype_metadata(self) -> Dict[str, object]:
        payload = {}
        for name, tensor in self.tensor_fields().items():
            if tensor is None:
                payload[name] = None
            else:
                payload[name] = {
                    "shape": list(tensor.shape),
                    "dtype": str(tensor.dtype),
                    "requires_grad": bool(tensor.requires_grad),
                }
        if self.ego_status is None:
            payload["ego_status"] = None
        else:
            payload["ego_status"] = {
                "shape": list(self.ego_status.shape),
                "dtype": str(self.ego_status.dtype),
                "requires_grad": bool(self.ego_status.requires_grad),
            }
        return payload

    def shapes_valid(self) -> bool:
        checks = []
        if self.plan_query is not None:
            checks.append(list(self.plan_query.shape)[-3:] == [1, 18, 256])
        if self.candidate_trajectory is not None:
            checks.append(list(self.candidate_trajectory.shape)[-4:] == [1, 18, 12, 2])
        if self.selected_planning is not None:
            checks.append(list(self.selected_planning.shape)[-2:] == [12, 2])
        if self.candidate_logits is not None:
            checks.append(list(self.candidate_logits.shape)[-2:] == [1, 18])
        if self.ego_status is not None:
            checks.append(list(self.ego_status.shape)[-1:] == [10])
        return all(checks) if checks else True


@dataclass
class HistoryRetrievalResult:
    entries_oldest_to_newest: List[CausalHistoryEntry]
    effective_history_len: int
    per_age_valid_newest_first: List[bool]
    stop_reason: Optional[str]
    audit: Dict[str, object]

    @property
    def memory_bypass(self) -> bool:
        return self.effective_history_len == 0


class CausalHistoryCache:
    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str], CausalHistoryEntry] = {}

    def clear(self) -> None:
        self._entries.clear()

    def write(self, entry: CausalHistoryEntry) -> None:
        self._entries[entry.key()] = entry

    def get(self, scene_token: str, sample_token: str) -> Optional[CausalHistoryEntry]:
        return self._entries.get((scene_token, sample_token))

    def __len__(self) -> int:
        return len(self._entries)


class CausalHistorySequenceRetriever:
    def __init__(
        self,
        cache: CausalHistoryCache,
        prev_token_getter: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.cache = cache
        self.prev_token_getter = prev_token_getter

    def retrieve(
        self,
        current_sample_token: str,
        current_prev_token: str,
        current_scene_token: str,
        current_rank_id: int,
        requested_history_len: int,
        current_timestamp: Optional[float] = None,
        current_causal_order: Optional[int] = None,
    ) -> HistoryRetrievalResult:
        if requested_history_len < 1:
            raise ValueError("requested_history_len must be >= 1")
        entries_newest_first: List[CausalHistoryEntry] = []
        valid_mask: List[bool] = []
        counters = {
            "requested_history_len": int(requested_history_len),
            "current_sample_token": current_sample_token,
            "current_prev_token": current_prev_token,
            "current_scene_token": current_scene_token,
            "current_rank_id": int(current_rank_id),
            "stop_reason": None,
            "checked_ages": [],
        }
        expected_token = current_prev_token
        newer_timestamp = current_timestamp
        for age in range(1, requested_history_len + 1):
            age_record = {"age": age, "expected_token": expected_token}
            if not expected_token:
                counters["stop_reason"] = "scene_boundary"
                age_record["valid"] = False
                age_record["reason"] = "scene_boundary"
                counters["checked_ages"].append(age_record)
                break
            entry = self.cache.get(current_scene_token, expected_token)
            if entry is None:
                counters["stop_reason"] = "missing_cache"
                age_record["valid"] = False
                age_record["reason"] = "missing_cache"
                counters["checked_ages"].append(age_record)
                break
            reason = self._entry_invalid_reason(
                entry,
                expected_token,
                current_scene_token,
                current_rank_id,
                newer_timestamp,
                current_causal_order,
            )
            if reason is not None:
                counters["stop_reason"] = reason
                age_record["valid"] = False
                age_record["reason"] = reason
                age_record["entry_token"] = entry.sample_token
                counters["checked_ages"].append(age_record)
                break
            expected_prev = None
            if self.prev_token_getter is not None:
                expected_prev = self.prev_token_getter(entry.sample_token)
                if expected_prev != entry.prev_token:
                    counters["stop_reason"] = "prev_token_mismatch"
                    age_record["valid"] = False
                    age_record["reason"] = "prev_token_mismatch"
                    age_record["entry_prev_token"] = entry.prev_token
                    age_record["expected_prev_token"] = expected_prev
                    counters["checked_ages"].append(age_record)
                    break
            entries_newest_first.append(entry)
            valid_mask.append(True)
            age_record.update(
                {
                    "valid": True,
                    "entry_token": entry.sample_token,
                    "entry_prev_token": entry.prev_token,
                    "entry_timestamp": entry.timestamp,
                    "entry_order": entry.cache_write_order,
                }
            )
            counters["checked_ages"].append(age_record)
            expected_token = entry.prev_token
            newer_timestamp = entry.timestamp
        stop_reason = counters["stop_reason"]
        result_entries = list(reversed(entries_newest_first))
        return HistoryRetrievalResult(
            entries_oldest_to_newest=result_entries,
            effective_history_len=len(result_entries),
            per_age_valid_newest_first=valid_mask
            + [False] * (requested_history_len - len(valid_mask)),
            stop_reason=stop_reason,
            audit=counters,
        )

    @staticmethod
    def _entry_invalid_reason(
        entry: CausalHistoryEntry,
        expected_token: str,
        current_scene_token: str,
        current_rank_id: int,
        newer_timestamp: Optional[float],
        current_causal_order: Optional[int],
    ) -> Optional[str]:
        if entry.sample_token != expected_token:
            return "sample_token_mismatch"
        if entry.scene_token != current_scene_token:
            return "scene_mismatch"
        if entry.rank_id != int(current_rank_id):
            return "rank_mismatch"
        if entry.history_source != "model_prediction":
            return "non_model_prediction_history"
        if newer_timestamp is not None and not (entry.timestamp < newer_timestamp):
            return "timestamp_reversal"
        if current_causal_order is not None and not (
            entry.cache_write_order < current_causal_order
        ):
            return "read_before_write_violation"
        if not entry.neural_tensors_detached():
            return "non_detached_tensor"
        if not entry.shapes_valid():
            return "shape_dtype_mismatch"
        return None


def stack_valid_entries_for_memory(
    entries_oldest_to_newest: Iterable[CausalHistoryEntry],
    field_name: str,
) -> Optional[torch.Tensor]:
    tensors = [getattr(entry, field_name) for entry in entries_oldest_to_newest]
    tensors = [tensor for tensor in tensors if tensor is not None]
    if not tensors:
        return None
    return torch.stack(tensors, dim=0)
