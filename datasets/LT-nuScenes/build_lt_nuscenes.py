#!/usr/bin/env python3
"""Audit and build the Longitudinal-Transition nuScenes validation subset.

The selection is deliberately model-output-free.  It uses only nuScenes ground-
truth ego increments, masks, scene identifiers, and timestamps.  The phase
definitions mirror StableDrive's MSTS supervision code:

* stationary: |v_y| < v_stop
* accelerating: a_y > a_thr and not stationary
* decelerating: a_y < -a_thr and not stationary
* cruising: every other valid horizon

LT-nuScenes is a single self-contained protocol. Whole selected scenes are
exported so that SMM's immediately preceding-cycle cache and TPC evaluation
retain their original temporal continuity. Each exported record also carries an
``lt_eval_target`` flag: metrics are accumulated only on clean longitudinal
transition records, while the remaining records provide temporal context.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


PHASE_NAMES = {
    0: "stationary",
    1: "accelerating",
    2: "cruising",
    3: "decelerating",
}
EVENT_NAMES = (
    "restart",
    "stop",
    "sustained_acceleration",
    "sustained_deceleration",
    "trend_reversal",
)


@dataclass(frozen=True)
class AuditConfig:
    dt: float = 0.5
    v_stop: float = 0.5
    a_thr: float = 0.5
    min_run: int = 2
    episode_gap: int = 2
    turning_deg: float = 15.0
    turning_displacement: float = 2.0
    lane_change_lateral: float = 1.5
    lane_change_forward: float = 5.0
    expected_scene_dt: float = 0.5
    # nuScenes key frames are nominally 2 Hz, but the annotation timestamps in
    # this converted validation file vary from about 0.40 s to 0.65 s.
    timestamp_tolerance: float = 0.20


@dataclass
class SampleEvent:
    original_index: int
    token: str
    scene_token: str
    timestamp: int
    valid_horizons: int
    phase_sequence: list[str]
    speed_min: float | None
    speed_max: float | None
    events: dict[str, list[int]]
    full_valid_gt: bool
    turning: bool
    lane_change: bool
    delta_yaw_deg: float | None
    displacement_6s: float
    lateral_displacement_6s: float | None
    forward_displacement_6s: float | None

    @property
    def has_event(self) -> bool:
        return any(self.events.values())

    @property
    def is_metric_target(self) -> bool:
        return (
            self.has_event
            and self.full_valid_gt
            and not self.turning
            and not self.lane_change
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-pkl", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--v-stop", type=float, default=0.5)
    parser.add_argument("--a-thr", type=float, default=0.5)
    parser.add_argument("--min-run", type=int, default=2)
    parser.add_argument("--episode-gap", type=int, default=2)
    parser.add_argument("--turning-deg", type=float, default=15.0)
    parser.add_argument("--turning-displacement", type=float, default=2.0)
    parser.add_argument("--lane-change-lateral", type=float, default=1.5)
    parser.add_argument("--lane-change-forward", type=float, default=5.0)
    parser.add_argument(
        "--selection-rule",
        choices=("none", "lt_v1", "balanced_v1", "inclusive", "strict"),
        default="none",
        help="Use none for a blind distribution audit before freezing a rule.",
    )
    parser.add_argument(
        "--export-subset",
        action="store_true",
        help="Write the selected PKL. Requires a non-none selection rule.",
    )
    return parser.parse_args()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_infos(path: Path) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    if isinstance(payload, dict) and isinstance(payload.get("infos"), list):
        return payload, payload["infos"], dict(payload.get("metadata", {}))
    if isinstance(payload, list):
        return payload, payload, {}
    raise TypeError("Expected a list or a dict containing an 'infos' list")


def phase_labels(
    increments: np.ndarray,
    masks: np.ndarray,
    cfg: AuditConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    increments = np.asarray(increments, dtype=np.float64)
    masks = np.asarray(masks).astype(bool)
    if increments.shape != (12, 2):
        raise ValueError(f"Expected gt_ego_fut_trajs (12, 2), got {increments.shape}")
    if masks.shape != (12,):
        raise ValueError(f"Expected gt_ego_fut_masks (12,), got {masks.shape}")

    vy = increments[:, 1] / cfg.dt
    ay = np.zeros_like(vy)
    ay[1:] = (vy[1:] - vy[:-1]) / cfg.dt
    labels = np.full(12, 2, dtype=np.int8)
    stationary = np.abs(vy) < cfg.v_stop
    labels[stationary] = 0
    labels[(ay > cfg.a_thr) & ~stationary] = 1
    labels[(ay < -cfg.a_thr) & ~stationary] = 3
    labels[~masks] = -1
    return labels, vy, ay


def lateral_semantics(
    increments: np.ndarray,
    masks: np.ndarray,
    cfg: AuditConfig,
) -> dict[str, Any]:
    """Mirror the old refined-subset Turning/Lane-change definitions."""
    increments = np.asarray(increments, dtype=np.float64)
    valid = np.asarray(masks) > 0.5
    cumulative = np.cumsum(increments, axis=0)
    valid_points = cumulative[valid]
    full_valid = bool(valid.all())

    displacement = float(np.linalg.norm(valid_points[-1])) if valid_points.size else 0.0
    lateral = float(valid_points[-1, 0]) if valid_points.size else None
    forward = float(valid_points[-1, 1]) if valid_points.size else None
    delta_yaw_deg: float | None = None
    if len(valid_points) >= 3:
        start_delta = valid_points[1] - valid_points[0]
        end_delta = valid_points[-1] - valid_points[-2]
        if np.linalg.norm(start_delta) >= 1e-6 and np.linalg.norm(end_delta) >= 1e-6:
            start_heading = float(np.arctan2(start_delta[1], start_delta[0]))
            end_heading = float(np.arctan2(end_delta[1], end_delta[0]))
            delta_yaw = (end_heading - start_heading + np.pi) % (2.0 * np.pi) - np.pi
            delta_yaw_deg = float(np.degrees(delta_yaw))

    turning = bool(
        full_valid
        and delta_yaw_deg is not None
        and displacement >= cfg.turning_displacement
        and abs(delta_yaw_deg) >= cfg.turning_deg
    )
    lane_change = bool(
        full_valid
        and lateral is not None
        and forward is not None
        and abs(lateral) >= cfg.lane_change_lateral
        and forward >= cfg.lane_change_forward
    )
    return {
        "full_valid_gt": full_valid,
        "turning": turning,
        "lane_change": lane_change,
        "delta_yaw_deg": delta_yaw_deg,
        "displacement_6s": displacement,
        "lateral_displacement_6s": lateral,
        "forward_displacement_6s": forward,
    }


def constant_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Return inclusive valid, same-label runs as (start, end, phase)."""
    runs: list[tuple[int, int, int]] = []
    start: int | None = None
    current: int | None = None
    for idx, raw_label in enumerate(labels.tolist()):
        label = int(raw_label)
        if label < 0:
            if start is not None and current is not None:
                runs.append((start, idx - 1, current))
            start = None
            current = None
            continue
        if start is None:
            start = idx
            current = label
        elif label != current:
            runs.append((start, idx - 1, int(current)))
            start = idx
            current = label
    if start is not None and current is not None:
        runs.append((start, len(labels) - 1, int(current)))
    return runs


def is_valid_block(labels: np.ndarray, start: int, length: int) -> bool:
    end = start + length
    return start >= 0 and end <= len(labels) and bool(np.all(labels[start:end] >= 0))


def detect_events(labels: np.ndarray, cfg: AuditConfig) -> dict[str, list[int]]:
    events = {name: [] for name in EVENT_NAMES}
    runs = constant_runs(labels)

    accel_runs = [run for run in runs if run[2] == 1 and run[1] - run[0] + 1 >= cfg.min_run]
    decel_runs = [run for run in runs if run[2] == 3 and run[1] - run[0] + 1 >= cfg.min_run]
    stationary_runs = [
        run for run in runs if run[2] == 0 and run[1] - run[0] + 1 >= cfg.min_run
    ]

    events["sustained_acceleration"] = [run[0] for run in accel_runs]
    events["sustained_deceleration"] = [run[0] for run in decel_runs]

    for start, end, _phase in stationary_runs:
        after_start = end + 1
        if is_valid_block(labels, after_start, cfg.min_run):
            after = labels[after_start : after_start + cfg.min_run]
            lookahead = labels[after_start : min(len(labels), after_start + cfg.min_run + 1)]
            if np.all(after != 0) and np.any(lookahead == 1):
                events["restart"].append(after_start)

        before_start = start - cfg.min_run
        if is_valid_block(labels, before_start, cfg.min_run):
            before = labels[before_start:start]
            lookback = labels[max(0, start - cfg.min_run - 1) : start]
            if np.all(before != 0) and np.any(lookback == 3):
                events["stop"].append(start)

    dynamic_runs = sorted(accel_runs + decel_runs, key=lambda run: run[0])
    for first, second in zip(dynamic_runs, dynamic_runs[1:]):
        if first[2] != second[2]:
            events["trend_reversal"].append(second[0])

    for name in EVENT_NAMES:
        events[name] = sorted(set(int(index) for index in events[name]))
    return events


def onset_bucket(horizon_index: int, dt: float) -> str:
    seconds = (horizon_index + 1) * dt
    if seconds <= 2.0:
        return "early_0.5_2.0s"
    if seconds <= 3.5:
        return "middle_2.5_3.5s"
    return "far_4.0_6.0s"


def merge_anchor_indices(indices: Iterable[int], max_gap: int) -> list[tuple[int, int]]:
    ordered = sorted(set(int(index) for index in indices))
    if not ordered:
        return []
    episodes: list[tuple[int, int]] = []
    start = ordered[0]
    previous = ordered[0]
    for index in ordered[1:]:
        if index - previous <= max_gap:
            previous = index
            continue
        episodes.append((start, previous))
        start = index
        previous = index
    episodes.append((start, previous))
    return episodes


def selection_passes(summary: dict[str, Any], rule: str) -> bool:
    stop_restart = int(summary["stop_restart_episodes"])
    accel_decel = int(summary["accel_decel_episodes"])
    total = int(summary["total_event_episodes"])
    event_samples = int(summary["event_samples"])
    onset_buckets = summary["onset_bucket_counts"]
    represented_buckets = sum(int(onset_buckets.get(name, 0)) > 0 for name in (
        "early_0.5_2.0s",
        "middle_2.5_3.5s",
        "far_4.0_6.0s",
    ))
    event_episodes = summary["event_episode_counts"]
    if rule == "lt_v1":
        # Scene membership is decided only after removing incomplete, turning,
        # and lane-change records. A retained scene must still contain the full
        # longitudinal-transition semantics and cover all horizon bands.
        target_event_counts = summary["metric_target_event_sample_counts"]
        target_buckets = summary["metric_target_onset_bucket_counts"]
        return (
            int(target_event_counts.get("stop", 0)) >= 1
            and int(target_event_counts.get("restart", 0)) >= 1
            and int(target_event_counts.get("sustained_acceleration", 0)) >= 1
            and int(target_event_counts.get("sustained_deceleration", 0)) >= 1
            and all(int(target_buckets.get(bucket, 0)) >= 1 for bucket in (
                "early_0.5_2.0s",
                "middle_2.5_3.5s",
                "far_4.0_6.0s",
            ))
        )
    if rule == "inclusive":
        return total >= 1
    if rule == "balanced_v1":
        return (
            (stop_restart >= 1 or accel_decel >= 2)
            and event_samples >= 2
            and represented_buckets >= 2
        )
    if rule == "strict":
        return (
            stop_restart >= 1
            and accel_decel >= 2
            and total >= 3
            and event_samples >= 4
            and represented_buckets == 3
        )
    if rule == "none":
        return False
    raise ValueError(f"Unknown selection rule: {rule}")


def summarize_rule(scene_summaries: list[dict[str, Any]], rule: str) -> dict[str, Any]:
    selected = [summary for summary in scene_summaries if selection_passes(summary, rule)]
    return {
        "rule": rule,
        "selected_scenes": len(selected),
        "selected_records": int(sum(summary["records"] for summary in selected)),
        "event_samples": int(sum(summary["event_samples"] for summary in selected)),
        "metric_target_records": int(
            sum(summary["metric_target_records"] for summary in selected)
        ),
        "event_episodes": int(sum(summary["total_event_episodes"] for summary in selected)),
        "event_type_episodes": {
            name: int(sum(summary["event_episode_counts"].get(name, 0) for summary in selected))
            for name in EVENT_NAMES
        },
        "onset_bucket_counts": {
            bucket: int(sum(summary["onset_bucket_counts"].get(bucket, 0) for summary in selected))
            for bucket in ("early_0.5_2.0s", "middle_2.5_3.5s", "far_4.0_6.0s")
        },
    }


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.export_subset and args.selection_rule == "none":
        raise SystemExit("--export-subset requires a non-none --selection-rule")
    cfg = AuditConfig(
        dt=args.dt,
        v_stop=args.v_stop,
        a_thr=args.a_thr,
        min_run=args.min_run,
        episode_gap=args.episode_gap,
        turning_deg=args.turning_deg,
        turning_displacement=args.turning_displacement,
        lane_change_lateral=args.lane_change_lateral,
        lane_change_forward=args.lane_change_forward,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_payload, infos, metadata = load_infos(args.input_pkl)
    source_sha256 = sha256_file(args.input_pkl)

    phase_counts: Counter[str] = Counter()
    event_sample_counts: Counter[str] = Counter()
    onset_counts: Counter[str] = Counter()
    schema_errors: list[dict[str, Any]] = []
    events_by_scene: dict[str, list[SampleEvent]] = defaultdict(list)
    all_sample_events: list[SampleEvent] = []
    lateral_counts: Counter[str] = Counter()

    for index, info in enumerate(infos):
        try:
            labels, vy, _ay = phase_labels(
                info["gt_ego_fut_trajs"], info["gt_ego_fut_masks"], cfg
            )
        except Exception as exc:  # keep a complete audit trail
            schema_errors.append({
                "index": index,
                "token": str(info.get("token", "")),
                "error": str(exc),
            })
            continue

        valid = labels >= 0
        for label in labels[valid].tolist():
            phase_counts[PHASE_NAMES[int(label)]] += 1
        events = detect_events(labels, cfg)
        semantics = lateral_semantics(
            info["gt_ego_fut_trajs"], info["gt_ego_fut_masks"], cfg
        )
        lateral_counts["full_valid_gt"] += int(semantics["full_valid_gt"])
        lateral_counts["turning"] += int(semantics["turning"])
        lateral_counts["lane_change"] += int(semantics["lane_change"])
        for event_name, onsets in events.items():
            if onsets:
                event_sample_counts[event_name] += 1
            for onset in onsets:
                onset_counts[onset_bucket(onset, cfg.dt)] += 1

        valid_speeds = vy[valid]
        sample_event = SampleEvent(
            original_index=index,
            token=str(info.get("token", "")),
            scene_token=str(info.get("scene_token", "")),
            timestamp=int(info.get("timestamp", 0)),
            valid_horizons=int(valid.sum()),
            phase_sequence=[PHASE_NAMES[int(label)] if label >= 0 else "invalid" for label in labels],
            speed_min=float(valid_speeds.min()) if valid_speeds.size else None,
            speed_max=float(valid_speeds.max()) if valid_speeds.size else None,
            events=events,
            **semantics,
        )
        lateral_counts["event_records"] += int(sample_event.has_event)
        lateral_counts["metric_target_records"] += int(sample_event.is_metric_target)
        all_sample_events.append(sample_event)
        events_by_scene[sample_event.scene_token].append(sample_event)

    scene_summaries: list[dict[str, Any]] = []
    continuity_gaps: list[dict[str, Any]] = []
    for scene_token, samples in events_by_scene.items():
        samples.sort(key=lambda sample: (sample.timestamp, sample.original_index))
        position_by_original_index = {
            sample.original_index: position for position, sample in enumerate(samples)
        }
        per_event_anchor_positions: dict[str, list[int]] = {name: [] for name in EVENT_NAMES}
        per_scene_onsets: Counter[str] = Counter()
        target_event_sample_counts: Counter[str] = Counter()
        target_onsets: Counter[str] = Counter()
        target_event_anchor_positions: dict[str, list[int]] = {
            name: [] for name in EVENT_NAMES
        }
        for sample in samples:
            if not sample.has_event:
                continue
            position = position_by_original_index[sample.original_index]
            for event_name, onsets in sample.events.items():
                if onsets:
                    per_event_anchor_positions[event_name].append(position)
                for onset in onsets:
                    per_scene_onsets[onset_bucket(onset, cfg.dt)] += 1
            if sample.is_metric_target:
                for event_name, onsets in sample.events.items():
                    if onsets:
                        target_event_sample_counts[event_name] += 1
                        target_event_anchor_positions[event_name].append(position)
                    for onset in onsets:
                        target_onsets[onset_bucket(onset, cfg.dt)] += 1

        episode_counts: dict[str, int] = {}
        episode_ranges: dict[str, list[list[int]]] = {}
        for event_name, positions in per_event_anchor_positions.items():
            episodes = merge_anchor_indices(positions, cfg.episode_gap)
            episode_counts[event_name] = len(episodes)
            episode_ranges[event_name] = [[start, end] for start, end in episodes]

        target_episode_counts: dict[str, int] = {}
        for event_name, positions in target_event_anchor_positions.items():
            target_episode_counts[event_name] = len(
                merge_anchor_indices(positions, cfg.episode_gap)
            )

        timestamps = [sample.timestamp for sample in samples]
        scene_gap_count = 0
        for left, right in zip(timestamps, timestamps[1:]):
            seconds = (right - left) / 1_000_000.0
            if abs(seconds - cfg.expected_scene_dt) > cfg.timestamp_tolerance:
                scene_gap_count += 1
                continuity_gaps.append({
                    "scene_token": scene_token,
                    "left_timestamp": left,
                    "right_timestamp": right,
                    "delta_seconds": seconds,
                })

        event_sample_count = sum(sample.has_event for sample in samples)
        metric_target_count = sum(sample.is_metric_target for sample in samples)
        stop_restart = episode_counts["stop"] + episode_counts["restart"]
        accel_decel = (
            episode_counts["sustained_acceleration"]
            + episode_counts["sustained_deceleration"]
        )
        total_episodes = sum(episode_counts.values())
        scene_summaries.append({
            "scene_token": scene_token,
            "records": len(samples),
            "event_samples": int(event_sample_count),
            "event_density": float(event_sample_count / len(samples)) if samples else 0.0,
            "metric_target_records": int(metric_target_count),
            "metric_target_density": (
                float(metric_target_count / len(samples)) if samples else 0.0
            ),
            "metric_target_event_sample_counts": dict(target_event_sample_counts),
            "metric_target_event_episode_counts": target_episode_counts,
            "metric_target_onset_bucket_counts": dict(target_onsets),
            "incomplete_event_records": int(
                sum(sample.has_event and not sample.full_valid_gt for sample in samples)
            ),
            "turning_event_records": int(
                sum(sample.has_event and sample.turning for sample in samples)
            ),
            "lane_change_event_records": int(
                sum(sample.has_event and sample.lane_change for sample in samples)
            ),
            "event_episode_counts": episode_counts,
            "event_episode_ranges": episode_ranges,
            "stop_restart_episodes": int(stop_restart),
            "accel_decel_episodes": int(accel_decel),
            "total_event_episodes": int(total_episodes),
            "onset_bucket_counts": dict(per_scene_onsets),
            "timestamp_gap_count": int(scene_gap_count),
        })

    scene_summaries.sort(
        key=lambda summary: (
            -summary["stop_restart_episodes"],
            -summary["accel_decel_episodes"],
            -summary["event_samples"],
            summary["scene_token"],
        )
    )
    phase_total = sum(phase_counts.values())
    rule_grid = {
        rule: summarize_rule(scene_summaries, rule)
        for rule in ("inclusive", "balanced_v1", "strict", "lt_v1")
    }
    audit_summary = {
        "name": "Longitudinal-Transition nuScenes blind data audit",
        "source_pkl": str(args.input_pkl.resolve()),
        "source_sha256": source_sha256,
        "config": asdict(cfg),
        "source_records": len(infos),
        "source_scenes": len(events_by_scene),
        "schema_error_count": len(schema_errors),
        "phase_counts": dict(phase_counts),
        "phase_ratios_over_valid_horizons": {
            name: count / phase_total if phase_total else 0.0
            for name, count in phase_counts.items()
        },
        "event_sample_counts": dict(event_sample_counts),
        "lateral_semantic_counts": dict(lateral_counts),
        "onset_bucket_counts": dict(onset_counts),
        "source_timestamp_gap_count": len(continuity_gaps),
        "candidate_rule_grid": rule_grid,
        "active_selection_rule": args.selection_rule,
    }

    write_json(args.output_dir / "audit_summary.json", audit_summary)
    write_json(args.output_dir / "schema_errors.json", schema_errors)
    write_json(args.output_dir / "continuity_gaps.json", continuity_gaps)
    write_json(args.output_dir / "scene_summary.json", scene_summaries)

    with (args.output_dir / "scene_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "scene_token",
            "records",
            "event_samples",
            "event_density",
            "metric_target_records",
            "metric_target_density",
            "stop_restart_episodes",
            "accel_decel_episodes",
            "total_event_episodes",
            "early_onsets",
            "middle_onsets",
            "far_onsets",
            "timestamp_gap_count",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for summary in scene_summaries:
            buckets = summary["onset_bucket_counts"]
            writer.writerow({
                "scene_token": summary["scene_token"],
                "records": summary["records"],
                "event_samples": summary["event_samples"],
                "event_density": f"{summary['event_density']:.6f}",
                "metric_target_records": summary["metric_target_records"],
                "metric_target_density": f"{summary['metric_target_density']:.6f}",
                "stop_restart_episodes": summary["stop_restart_episodes"],
                "accel_decel_episodes": summary["accel_decel_episodes"],
                "total_event_episodes": summary["total_event_episodes"],
                "early_onsets": buckets.get("early_0.5_2.0s", 0),
                "middle_onsets": buckets.get("middle_2.5_3.5s", 0),
                "far_onsets": buckets.get("far_4.0_6.0s", 0),
                "timestamp_gap_count": summary["timestamp_gap_count"],
            })

    with (args.output_dir / "event_samples.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "original_index",
            "token",
            "scene_token",
            "timestamp",
            "valid_horizons",
            "full_valid_gt",
            "turning",
            "lane_change",
            "lt_eval_target",
            "delta_yaw_deg",
            "displacement_6s",
            "lateral_displacement_6s",
            "forward_displacement_6s",
            "speed_min",
            "speed_max",
            *EVENT_NAMES,
            "phase_sequence",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in all_sample_events:
            if not sample.has_event:
                continue
            writer.writerow({
                "original_index": sample.original_index,
                "token": sample.token,
                "scene_token": sample.scene_token,
                "timestamp": sample.timestamp,
                "valid_horizons": sample.valid_horizons,
                "full_valid_gt": sample.full_valid_gt,
                "turning": sample.turning,
                "lane_change": sample.lane_change,
                "lt_eval_target": sample.is_metric_target,
                "delta_yaw_deg": (
                    "" if sample.delta_yaw_deg is None else f"{sample.delta_yaw_deg:.6f}"
                ),
                "displacement_6s": f"{sample.displacement_6s:.6f}",
                "lateral_displacement_6s": (
                    "" if sample.lateral_displacement_6s is None
                    else f"{sample.lateral_displacement_6s:.6f}"
                ),
                "forward_displacement_6s": (
                    "" if sample.forward_displacement_6s is None
                    else f"{sample.forward_displacement_6s:.6f}"
                ),
                "speed_min": "" if sample.speed_min is None else f"{sample.speed_min:.6f}",
                "speed_max": "" if sample.speed_max is None else f"{sample.speed_max:.6f}",
                **{name: ";".join(map(str, sample.events[name])) for name in EVENT_NAMES},
                "phase_sequence": ";".join(sample.phase_sequence),
            })

    selected_scenes = {
        summary["scene_token"]
        for summary in scene_summaries
        if selection_passes(summary, args.selection_rule)
    }
    selected_event_samples = [
        sample for sample in all_sample_events
        if sample.scene_token in selected_scenes and sample.has_event
    ]
    metric_target_samples = [
        sample for sample in all_sample_events
        if sample.scene_token in selected_scenes and sample.is_metric_target
    ]
    manifest = {
        "subset_name": "Longitudinal-Transition nuScenes",
        "short_name": "LT-nuScenes",
        "selection_uses_model_outputs": False,
        "source_sha256": source_sha256,
        "selection_rule": args.selection_rule,
        "selection_rule_definition": {
            "lt_v1": {
                "record_target_rule": {
                    "longitudinal_event_required": True,
                    "complete_6s_gt_required": True,
                    "turning_excluded": {
                        "absolute_delta_yaw_deg_min": cfg.turning_deg,
                        "displacement_6s_m_min": cfg.turning_displacement,
                    },
                    "lane_change_excluded": {
                        "absolute_lateral_displacement_6s_m_min": cfg.lane_change_lateral,
                        "forward_displacement_6s_m_min": cfg.lane_change_forward,
                    },
                },
                "scene_rule": {
                    "required_target_event_types": [
                        "restart",
                        "stop",
                        "sustained_acceleration",
                        "sustained_deceleration",
                    ],
                    "required_target_onset_buckets": [
                        "early_0.5_2.0s",
                        "middle_2.5_3.5s",
                        "far_4.0_6.0s",
                    ],
                },
            }
        }.get(args.selection_rule),
        "config": asdict(cfg),
        "evaluation_protocol": {
            "inference_scope": "all records from every selected scene",
            "l2_boxcol_scope": "records with lt_eval_target=true",
            "tpc_scope": "records with lt_tpc_target=true",
            "cache_reset": "scene boundary only",
            "runtime_order": (
                "ascending timestamp as applied by NuScenes3DDataset.load_annotations; "
                "align full-validation predictions by sample token"
            ),
            "non_target_records_are_context_only": True,
            "scene_first_metric_targets_are_excluded_from_tpc": True,
        },
        "selected_scene_tokens": sorted(selected_scenes),
        "selected_record_tokens": [
            str(info.get("token", ""))
            for info in infos
            if str(info.get("scene_token", "")) in selected_scenes
        ],
        "metric_target_tokens": [sample.token for sample in metric_target_samples],
        "metric_target_annotations": [
            {
                "token": sample.token,
                "scene_token": sample.scene_token,
                "events": sample.events,
                "delta_yaw_deg": sample.delta_yaw_deg,
                "displacement_6s": sample.displacement_6s,
                "lateral_displacement_6s": sample.lateral_displacement_6s,
                "forward_displacement_6s": sample.forward_displacement_6s,
            }
            for sample in metric_target_samples
        ],
        "all_longitudinal_event_tokens_in_selected_scenes": [
            sample.token for sample in selected_event_samples
        ],
    }
    write_json(args.output_dir / "lt_nuscenes_manifest.json", manifest)

    if args.export_subset:
        event_by_index = {sample.original_index: sample for sample in all_sample_events}
        selected_infos = []
        seen_selected_scenes: set[str] = set()
        tpc_target_tokens: list[str] = []
        for index, info in enumerate(infos):
            scene_token = str(info.get("scene_token", ""))
            if scene_token not in selected_scenes:
                continue
            selected_info = dict(info)
            sample = event_by_index[index]
            selected_info["lt_eval_target"] = bool(sample.is_metric_target)
            selected_info["lt_tpc_target"] = bool(
                sample.is_metric_target and scene_token in seen_selected_scenes
            )
            selected_info["lt_event_types"] = [
                name for name in EVENT_NAMES if sample.events[name]
            ] if sample.is_metric_target else []
            selected_info["lt_subset_version"] = "v1"
            selected_infos.append(selected_info)
            seen_selected_scenes.add(scene_token)
            if selected_info["lt_tpc_target"]:
                tpc_target_tokens.append(str(selected_info["token"]))
        manifest["tpc_target_tokens"] = tpc_target_tokens
        write_json(args.output_dir / "lt_nuscenes_manifest.json", manifest)
        subset_metadata = dict(metadata)
        subset_metadata["lt_nuscenes"] = {
            "version": "v1",
            "metric_target_field": "lt_eval_target",
            "tpc_target_field": "lt_tpc_target",
            "metric_target_records": len(metric_target_samples),
            "tpc_target_records": len(tpc_target_tokens),
            "inference_records": len(selected_infos),
        }
        if isinstance(source_payload, dict):
            subset_payload = dict(source_payload)
            subset_payload["infos"] = selected_infos
            subset_payload["metadata"] = subset_metadata
        else:
            subset_payload = selected_infos
        subset_path = args.output_dir / "nuscenes_infos_val_lt.pkl"
        with subset_path.open("wb") as handle:
            pickle.dump(subset_payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        subset_sha256 = sha256_file(subset_path)
        export_summary = {
            "subset_path": str(subset_path.resolve()),
            "subset_sha256": subset_sha256,
            "selected_scenes": len(selected_scenes),
            "selected_records": len(selected_infos),
            "metric_target_records": len(metric_target_samples),
            "tpc_target_records": len(tpc_target_tokens),
            "context_only_records": len(selected_infos) - len(metric_target_samples),
            "all_records_belong_to_selected_scenes": all(
                str(info.get("scene_token", "")) in selected_scenes for info in selected_infos
            ),
            "source_order_preserved": [
                str(info.get("token", "")) for info in selected_infos
            ] == [
                str(info.get("token", ""))
                for info in infos
                if str(info.get("scene_token", "")) in selected_scenes
            ],
            "annotation_file_order_preserved": True,
            "official_runtime_order": (
                "ascending timestamp as applied by NuScenes3DDataset.load_annotations"
            ),
            "runtime_alignment": "sample-token mapping to full-validation prediction order",
            "metric_target_flags_match_manifest": {
                str(info["token"])
                for info in selected_infos if info["lt_eval_target"]
            } == {sample.token for sample in metric_target_samples},
            "tpc_target_flags_match_manifest": {
                str(info["token"])
                for info in selected_infos if info["lt_tpc_target"]
            } == set(tpc_target_tokens),
        }
        write_json(args.output_dir / "export_summary.json", export_summary)

    report_lines = [
        "# LT-nuScenes blind data audit",
        "",
        f"- Source records: `{len(infos)}`",
        f"- Source scenes: `{len(events_by_scene)}`",
        f"- Source SHA-256: `{source_sha256}`",
        f"- Schema errors: `{len(schema_errors)}`",
        f"- Timestamp continuity gaps: `{len(continuity_gaps)}`",
        f"- Phase thresholds: `dt={cfg.dt}`, `v_stop={cfg.v_stop}`, `a_thr={cfg.a_thr}`",
        f"- Persistence rule: `{cfg.min_run}` consecutive horizons",
        "- Model outputs used for selection: `No`",
        "",
        "## Phase distribution over valid horizons",
        "",
        "| Phase | Count | Ratio |",
        "|---|---:|---:|",
    ]
    for name in PHASE_NAMES.values():
        count = phase_counts.get(name, 0)
        ratio = count / phase_total if phase_total else 0.0
        report_lines.append(f"| {name} | {count} | {ratio:.6f} |")
    report_lines.extend([
        "",
        "## Event-bearing samples",
        "",
        "| Event | Samples |",
        "|---|---:|",
    ])
    for name in EVENT_NAMES:
        report_lines.append(f"| {name} | {event_sample_counts.get(name, 0)} |")
    report_lines.extend([
        "",
        "## Candidate scene-selection rules",
        "",
        "| Rule | Scenes | Inference records | Event records | Metric targets |",
        "|---|---:|---:|---:|---:|",
    ])
    for rule in ("inclusive", "balanced_v1", "strict", "lt_v1"):
        summary = rule_grid[rule]
        report_lines.append(
            f"| {rule} | {summary['selected_scenes']} | {summary['selected_records']} | "
            f"{summary['event_samples']} | {summary['metric_target_records']} |"
        )
    report_lines.extend([""])
    if args.selection_rule == "none":
        report_lines.extend([
            "`none` is the required first-pass audit mode. A selection rule must be frozen "
            "from these data-only distributions before any model inference on LT-nuScenes.",
            "",
        ])
    else:
        active_summary = rule_grid[args.selection_rule]
        report_lines.extend([
            "## Frozen active rule",
            "",
            f"- Rule: `{args.selection_rule}`",
            f"- Selected scenes: `{active_summary['selected_scenes']}`",
            f"- Selected records: `{active_summary['selected_records']}`",
            f"- Metric target records: `{active_summary['metric_target_records']}`",
            "- Inference uses every record in each selected scene.",
            "- L2 and BoxCol use records whose `lt_eval_target` field is true.",
            "- TPC additionally requires `lt_tpc_target=true`.",
            "",
        ])
    (args.output_dir / "audit_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    print(json.dumps(audit_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
