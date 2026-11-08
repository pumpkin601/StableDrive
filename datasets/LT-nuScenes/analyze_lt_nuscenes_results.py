#!/usr/bin/env python3
"""Independently re-aggregate and bootstrap frozen LT-nuScenes results."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


HORIZON_STEPS = {f"{second}s": second * 2 for second in range(1, 7)}
MODEL_ORDER = [
    "Base Planner (A0)",
    "SMM Planner",
    "MSTS-Trained Planner",
    "StableDrive",
]
METRIC_KEYS = {
    "L2": "L2_raw_per_step",
    "BoxCol_fraction": "BoxCol_raw_per_step",
    "TPC": "TPC_raw_per_step",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20260803)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def horizon_values(raw_per_step: np.ndarray) -> dict:
    return {
        horizon: float(raw_per_step[:steps].mean())
        for horizon, steps in HORIZON_STEPS.items()
    }


def independent_reaggregation(per_record: list, formal_metrics: dict) -> dict:
    output = {"status": "PASS", "models": {}}
    for model in MODEL_ORDER:
        model_output = {}
        for metric, record_key in METRIC_KEYS.items():
            arrays = []
            for row in per_record:
                value = row["models"][model].get(record_key)
                if value is not None:
                    arrays.append(np.asarray(value, dtype=np.float64))
            matrix = np.stack(arrays, axis=0)
            mean_per_step = matrix.mean(axis=0)
            horizons = horizon_values(mean_per_step)
            formal_key = metric
            expected = formal_metrics[model][formal_key]
            if matrix.shape[0] != expected["records"]:
                raise RuntimeError(f"record count mismatch: {model}/{metric}")
            if not np.allclose(
                mean_per_step,
                np.asarray(expected["raw_per_step"], dtype=np.float64),
                rtol=0.0,
                atol=1e-12,
            ):
                raise RuntimeError(f"per-step reaggregation mismatch: {model}/{metric}")
            for horizon, value in horizons.items():
                if not np.isclose(
                    value,
                    expected["horizons"][horizon],
                    rtol=0.0,
                    atol=1e-12,
                ):
                    raise RuntimeError(f"horizon reaggregation mismatch: {model}/{metric}/{horizon}")
            model_output[metric] = {
                "records": int(matrix.shape[0]),
                "raw_per_step": mean_per_step.tolist(),
                "horizons": horizons,
            }
        output["models"][model] = model_output
    return output


def scene_contributions(per_record: list) -> dict:
    output = {}
    scenes = sorted({row["scene_token"] for row in per_record})
    for model in MODEL_ORDER:
        output[model] = {}
        for scene in scenes:
            rows = [row for row in per_record if row["scene_token"] == scene]
            metrics = {}
            for metric, record_key in METRIC_KEYS.items():
                arrays = [
                    np.asarray(row["models"][model][record_key], dtype=np.float64)
                    for row in rows
                    if record_key in row["models"][model]
                ]
                matrix = np.stack(arrays, axis=0)
                metrics[metric] = {
                    "count": int(matrix.shape[0]),
                    "sum": matrix.sum(axis=0),
                }
            output[model][scene] = metrics
    return output


def aggregate_resampled(contributions: dict, model: str, sampled_scenes: np.ndarray, metric: str, steps: int) -> float:
    raw_sum = np.zeros(12, dtype=np.float64)
    total = 0
    for scene in sampled_scenes.tolist():
        item = contributions[model][scene][metric]
        raw_sum += item["sum"]
        total += item["count"]
    if total <= 0:
        raise RuntimeError(f"zero bootstrap denominator: {model}/{metric}")
    return float((raw_sum / float(total))[:steps].mean())


def paired_scene_bootstrap(per_record: list, replicates: int, seed: int) -> dict:
    if replicates <= 0:
        raise RuntimeError("bootstrap replicates must be positive")
    contributions = scene_contributions(per_record)
    scenes = np.asarray(sorted(contributions[MODEL_ORDER[0]].keys()))
    if len(scenes) != 16:
        raise RuntimeError(f"scene count mismatch for bootstrap: {len(scenes)}")
    rng = np.random.default_rng(seed)
    output = {
        "status": "PASS",
        "method": "paired scene-level nonparametric bootstrap with count-weighted metric aggregation",
        "seed": seed,
        "replicates": replicates,
        "scene_count": int(len(scenes)),
        "comparisons": {},
    }
    base = MODEL_ORDER[0]
    for candidate in MODEL_ORDER[1:]:
        candidate_output = {}
        for metric in METRIC_KEYS:
            metric_output = {}
            for horizon, steps in HORIZON_STEPS.items():
                differences = np.empty(replicates, dtype=np.float64)
                relative_improvements = np.empty(replicates, dtype=np.float64)
                relative_improvements.fill(np.nan)
                for index in range(replicates):
                    sampled = rng.choice(scenes, size=len(scenes), replace=True)
                    base_value = aggregate_resampled(contributions, base, sampled, metric, steps)
                    candidate_value = aggregate_resampled(
                        contributions, candidate, sampled, metric, steps
                    )
                    differences[index] = candidate_value - base_value
                    if base_value != 0.0:
                        relative_improvements[index] = 100.0 * (
                            base_value - candidate_value
                        ) / base_value
                finite_relative = relative_improvements[np.isfinite(relative_improvements)]
                metric_output[horizon] = {
                    "candidate_minus_base_mean": float(differences.mean()),
                    "candidate_minus_base_ci95": [
                        float(np.quantile(differences, 0.025)),
                        float(np.quantile(differences, 0.975)),
                    ],
                    "probability_candidate_lower": float(np.mean(differences < 0.0)),
                    "relative_improvement_percent_mean": (
                        float(finite_relative.mean()) if finite_relative.size else None
                    ),
                    "relative_improvement_percent_ci95": (
                        [
                            float(np.quantile(finite_relative, 0.025)),
                            float(np.quantile(finite_relative, 0.975)),
                        ]
                        if finite_relative.size
                        else None
                    ),
                }
            candidate_output[metric] = metric_output
        output["comparisons"][candidate] = candidate_output
    return output


def main() -> int:
    args = parse_args()
    result_dir = Path(args.result_dir).resolve()
    metrics_path = result_dir / "lt_nuscenes_metrics.json"
    records_path = result_dir / "per_record_metrics.json"
    formal = json.loads(metrics_path.read_text(encoding="utf-8"))
    per_record = json.loads(records_path.read_text(encoding="utf-8"))
    if formal.get("status") != "PASS" or len(per_record) != 189:
        raise RuntimeError("formal LT result is incomplete")
    reaggregation = independent_reaggregation(per_record, formal["metrics"])
    reaggregation["inputs"] = {
        "lt_nuscenes_metrics_sha256": sha256_file(metrics_path),
        "per_record_metrics_sha256": sha256_file(records_path),
    }
    bootstrap = paired_scene_bootstrap(
        per_record,
        args.bootstrap_replicates,
        args.seed,
    )
    bootstrap["inputs"] = dict(reaggregation["inputs"])
    atomic_json(result_dir / "independent_reaggregation_audit.json", reaggregation)
    atomic_json(result_dir / "paired_scene_bootstrap.json", bootstrap)
    print(
        json.dumps(
            {
                "status": "PASS",
                "records": len(per_record),
                "bootstrap_replicates": args.bootstrap_replicates,
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
