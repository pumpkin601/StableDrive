#!/usr/bin/env python3
"""Fail-closed offline evaluation for the frozen LT-nuScenes v1 subset.

The four model predictions were generated over the complete nuScenes validation
sequence.  This evaluator extracts the 642 records belonging to the 16 frozen
LT scenes, keeps every record as temporal context, and accumulates metrics only
where the annotation flags request it:

* ``lt_eval_target`` for L2 and BoxCol (189 records);
* ``lt_tpc_target`` for TPC (176 records).

No training, inference, checkpoint selection, or subset selection is performed
by this script.  Every large source artifact is opened read-only and audited.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


EXPECTED_SOURCE_RECORDS = 6019
EXPECTED_LT_RECORDS = 642
EXPECTED_LT_SCENES = 16
EXPECTED_EVAL_TARGETS = 189
EXPECTED_TPC_TARGETS = 176
EXPECTED_SOURCE_SHA256 = "c77120d97b683b7285e4a3440db52e7d806f054464f6747cf31b99f90bc6653a"
EXPECTED_LT_SHA256 = "1ffc868c0afe9f1149657b88f2e019f511c86e5f6cc6add1de7be15d82e03d87"
HORIZON_STEPS = {f"{second}s": second * 2 for second in range(1, 7)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-loop-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-pkl", required=True)
    parser.add_argument("--lt-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help="NAME|RESULTS_PKL|CHECKPOINT|EXPECTED_CHECKPOINT_SHA256",
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--preflight-only", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_pickle(path: Path):
    with path.open("rb") as handle:
        return pickle.load(handle)


class CPUUnpickler(pickle.Unpickler):
    """Load torch storages on CPU even when the pickle records a CUDA device."""

    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu")
        return super().find_class(module, name)


def read_results_on_cpu(path: Path):
    with path.open("rb") as handle:
        return CPUUnpickler(handle).load()


def infos_from_pickle(path: Path) -> list:
    payload = read_pickle(path)
    infos = payload["infos"] if isinstance(payload, dict) else payload
    if not isinstance(infos, list):
        raise RuntimeError(f"annotation infos are not a list: {path}")
    return infos


def atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"stale temporary output exists: {tmp}")
    with tmp.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    if tmp.exists():
        raise RuntimeError(f"stale temporary output exists: {tmp}")
    np.savez_compressed(str(tmp), **arrays)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_variants(raw_specs: list) -> list:
    variants = []
    seen = set()
    for raw in raw_specs:
        parts = raw.split("|", 3)
        if len(parts) != 4:
            raise RuntimeError(f"invalid --variant value: {raw}")
        name, result_path, checkpoint_path, expected_checkpoint_sha = parts
        if not name or name in seen:
            raise RuntimeError(f"empty or duplicate variant name: {name!r}")
        seen.add(name)
        variants.append(
            {
                "name": name,
                "results": Path(result_path).resolve(),
                "checkpoint": Path(checkpoint_path).resolve(),
                "expected_checkpoint_sha256": expected_checkpoint_sha,
            }
        )
    expected_names = [
        "Base Planner (A0)",
        "SMM Planner",
        "MSTS-Trained Planner",
        "StableDrive",
    ]
    actual_names = [item["name"] for item in variants]
    if actual_names != expected_names:
        raise RuntimeError(f"variant order/identity mismatch: {actual_names}")
    return variants


def validate_annotations(source_path: Path, lt_path: Path):
    source_sha = sha256_file(source_path)
    lt_sha = sha256_file(lt_path)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source annotation SHA mismatch: {source_sha}")
    if lt_sha != EXPECTED_LT_SHA256:
        raise RuntimeError(f"LT annotation SHA mismatch: {lt_sha}")

    source_infos_file_order = infos_from_pickle(source_path)
    lt_infos_file_order = infos_from_pickle(lt_path)
    if len(source_infos_file_order) != EXPECTED_SOURCE_RECORDS:
        raise RuntimeError(f"source record count mismatch: {len(source_infos_file_order)}")
    if len(lt_infos_file_order) != EXPECTED_LT_RECORDS:
        raise RuntimeError(f"LT record count mismatch: {len(lt_infos_file_order)}")

    # NuScenes3DDataset sorts its annotation records by timestamp in
    # load_annotations().  Full-validation results.pkl therefore follows this
    # official runtime order, which can differ from the raw pickle list order.
    source_file_tokens = [str(info["token"]) for info in source_infos_file_order]
    lt_file_tokens = [str(info["token"]) for info in lt_infos_file_order]
    source_infos = sorted(source_infos_file_order, key=lambda info: info["timestamp"])
    lt_infos = sorted(lt_infos_file_order, key=lambda info: info["timestamp"])
    source_tokens = [str(info["token"]) for info in source_infos]
    lt_tokens = [str(info["token"]) for info in lt_infos]
    if len(set(source_tokens)) != len(source_tokens):
        raise RuntimeError("duplicate token in source annotation")
    if len(set(lt_tokens)) != len(lt_tokens):
        raise RuntimeError("duplicate token in LT annotation")
    source_index = {token: index for index, token in enumerate(source_tokens)}
    if any(token not in source_index for token in lt_tokens):
        raise RuntimeError("LT token absent from source annotation")
    selected_source_indices = [source_index[token] for token in lt_tokens]
    if selected_source_indices != sorted(selected_source_indices):
        raise RuntimeError("LT runtime order does not preserve official source runtime order")

    scene_tokens = [str(info["scene_token"]) for info in lt_infos]
    if len(set(scene_tokens)) != EXPECTED_LT_SCENES:
        raise RuntimeError(f"LT scene count mismatch: {len(set(scene_tokens))}")
    eval_count = sum(bool(info.get("lt_eval_target")) for info in lt_infos)
    tpc_count = sum(bool(info.get("lt_tpc_target")) for info in lt_infos)
    if eval_count != EXPECTED_EVAL_TARGETS or tpc_count != EXPECTED_TPC_TARGETS:
        raise RuntimeError(f"LT target count mismatch: eval={eval_count}, tpc={tpc_count}")

    previous_scene = None
    seen_scenes = set()
    scene_first = True
    for info in lt_infos:
        scene = str(info["scene_token"])
        if scene != previous_scene:
            if scene in seen_scenes:
                raise RuntimeError(f"LT scene is not contiguous: {scene}")
            seen_scenes.add(scene)
            previous_scene = scene
            scene_first = True
        if info.get("lt_tpc_target") and scene_first:
            raise RuntimeError("scene-first record cannot be an LT TPC target")
        if info.get("lt_tpc_target") and not info.get("lt_eval_target"):
            raise RuntimeError("LT TPC target must also be an LT metric target")
        if info.get("lt_eval_target"):
            mask = np.asarray(info["gt_ego_fut_masks"]).reshape(-1)[:12]
            if mask.shape != (12,) or not mask.astype(bool).all():
                raise RuntimeError(f"incomplete metric target: {info['token']}")
        scene_first = False

    return {
        "source_sha256": source_sha,
        "lt_sha256": lt_sha,
        "source_infos": source_infos,
        "lt_infos": lt_infos,
        "source_file_order_differs_from_runtime_order": source_file_tokens != source_tokens,
        "lt_file_order_differs_from_runtime_order": lt_file_tokens != lt_tokens,
        "runtime_order": "ascending timestamp, matching NuScenes3DDataset.load_annotations",
        "source_index": source_index,
        "selected_source_indices": selected_source_indices,
    }


def validate_variant_files(variants: list) -> list:
    audited = []
    for variant in variants:
        results_path = variant["results"]
        checkpoint_path = variant["checkpoint"]
        if not results_path.is_file() or not checkpoint_path.is_file():
            raise RuntimeError(f"missing artifact for {variant['name']}")
        checkpoint_sha = sha256_file(checkpoint_path)
        if checkpoint_sha != variant["expected_checkpoint_sha256"]:
            raise RuntimeError(
                f"checkpoint SHA mismatch for {variant['name']}: {checkpoint_sha}"
            )
        audited.append(
            {
                **variant,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_size_bytes": checkpoint_path.stat().st_size,
                "results_size_bytes": results_path.stat().st_size,
                "results_mtime_ns_before": results_path.stat().st_mtime_ns,
            }
        )
    return audited


def validate_dataset_runtime_order(
    open_loop_root: Path,
    config_path: Path,
    lt_path: Path,
    lt_infos: list,
) -> None:
    sys.path.insert(0, str(open_loop_root))
    os.chdir(str(open_loop_root))
    from mmcv import Config
    from mmdet.datasets import build_dataset

    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(str(config_path))
    eval_config = cfg.eval_config.copy()
    eval_config["ann_file"] = str(lt_path)
    dataset = build_dataset(eval_config)
    if len(dataset) != EXPECTED_LT_RECORDS:
        raise RuntimeError(f"built LT dataset length mismatch: {len(dataset)}")
    dataset_tokens = [str(info["token"]) for info in dataset.data_infos]
    expected_tokens = [str(info["token"]) for info in lt_infos]
    if dataset_tokens != expected_tokens:
        raise RuntimeError("official LT runtime order differs from the audited timestamp order")


def compact_predictions(
    variants: list,
    selected_source_indices: list,
    lt_infos: list,
    out_dir: Path,
) -> tuple:
    tokens = np.asarray([str(info["token"]) for info in lt_infos], dtype="S32")
    scene_tokens = np.asarray([str(info["scene_token"]) for info in lt_infos], dtype="S32")
    compact = {}
    audit_rows = []
    for variant in variants:
        started = time.time()
        results_path = variant["results"]
        before = results_path.stat()
        results = read_results_on_cpu(results_path)
        if not isinstance(results, list) or len(results) != EXPECTED_SOURCE_RECORDS:
            raise RuntimeError(
                f"result length/type mismatch for {variant['name']}: "
                f"type={type(results).__name__}, len={len(results)}"
            )

        planning = np.empty((EXPECTED_LT_RECORDS, 12, 2), dtype=np.float32)
        for compact_index, source_index in enumerate(selected_source_indices):
            sample = results[source_index]
            if not isinstance(sample, dict) or "img_bbox" not in sample:
                raise RuntimeError(f"invalid result sample for {variant['name']} at {source_index}")
            value = sample["img_bbox"].get("final_planning")
            if value is None:
                raise RuntimeError(f"missing final_planning for {variant['name']} at {source_index}")
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            value = np.asarray(value, dtype=np.float32)
            if value.shape != (12, 2) or not np.isfinite(value).all():
                raise RuntimeError(
                    f"invalid final_planning for {variant['name']} at {source_index}: {value.shape}"
                )
            planning[compact_index] = value

        del results
        gc.collect()
        after = results_path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source result changed during read: {variant['name']}")

        safe_name = (
            variant["name"].lower()
            .replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("-", "_")
        )
        compact_path = out_dir / f"{safe_name}_lt_predictions.npz"
        atomic_savez(
            compact_path,
            token=tokens,
            scene_token=scene_tokens,
            final_planning=planning,
        )
        compact[variant["name"]] = planning
        audit_rows.append(
            {
                "name": variant["name"],
                "results": str(results_path),
                "results_size_bytes": before.st_size,
                "results_mtime_ns": before.st_mtime_ns,
                "results_sha256": sha256_file(results_path),
                "compact_predictions": str(compact_path),
                "compact_predictions_sha256": sha256_file(compact_path),
                "compact_prediction_shape": list(planning.shape),
                "elapsed_seconds": time.time() - started,
            }
        )
    return compact, audit_rows


def prefix_horizons(raw_sum: np.ndarray, total: int) -> dict:
    if total <= 0:
        raise RuntimeError("cannot summarize a metric with zero records")
    per_step = np.asarray(raw_sum, dtype=np.float64) / float(total)
    horizons = {
        horizon: float(per_step[:steps].mean())
        for horizon, steps in HORIZON_STEPS.items()
    }
    return {
        "records": int(total),
        "raw_per_step": [float(value) for value in per_step.tolist()],
        "horizons": horizons,
        "avg_1_6_horizons": float(np.mean(list(horizons.values()))),
        "avg_4_6_horizons": float(
            np.mean([horizons["4s"], horizons["5s"], horizons["6s"]])
        ),
    }


def evaluate_metrics(
    open_loop_root: Path,
    config_path: Path,
    lt_path: Path,
    lt_infos: list,
    compact: dict,
    workers: int,
) -> tuple:
    sys.path.insert(0, str(open_loop_root))
    os.chdir(str(open_loop_root))
    from mmcv import Config
    from mmdet.datasets import build_dataloader, build_dataset

    import projects.mmdet3d_plugin  # noqa: F401
    from projects.mmdet3d_plugin.datasets.evaluation.planning.planning_eval_roboAD_6s import (
        PlanningMetric,
    )

    cfg = Config.fromfile(str(config_path))
    eval_config = cfg.eval_config.copy()
    eval_config["ann_file"] = str(lt_path)
    dataset = build_dataset(eval_config)
    if len(dataset) != EXPECTED_LT_RECORDS:
        raise RuntimeError(f"built LT dataset length mismatch: {len(dataset)}")
    dataset_tokens = [str(info["token"]) for info in dataset.data_infos]
    expected_tokens = [str(info["token"]) for info in lt_infos]
    if dataset_tokens != expected_tokens:
        raise RuntimeError("built LT dataset order differs from frozen annotation order")

    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=max(0, int(workers)),
        shuffle=False,
        dist=False,
    )
    metric_impl = PlanningMetric()
    accumulators = {
        name: {
            "L2": np.zeros(12, dtype=np.float64),
            "obj_box_col": np.zeros(12, dtype=np.float64),
            "TPC": np.zeros(12, dtype=np.float64),
            "eval_count": 0,
            "tpc_count": 0,
        }
        for name in compact
    }
    previous = {name: None for name in compact}
    previous_scene = None
    per_record = []
    scene_sums = {
        name: defaultdict(
            lambda: {
                "L2": np.zeros(12, dtype=np.float64),
                "obj_box_col": np.zeros(12, dtype=np.float64),
                "TPC": np.zeros(12, dtype=np.float64),
                "eval_count": 0,
                "tpc_count": 0,
            }
        )
        for name in compact
    }

    processed = 0
    for index, data in enumerate(dataloader):
        info = lt_infos[index]
        token = str(info["token"])
        scene = str(info["scene_token"])
        batch_token = str(data["img_metas"].data[0][0]["token"])
        if batch_token != token:
            raise RuntimeError(f"dataloader token mismatch at {index}: {batch_token} != {token}")
        if scene != previous_scene:
            previous = {name: None for name in compact}
            previous_scene = scene

        gt = data["gt_ego_fut_trajs"].cumsum(dim=-2).unsqueeze(1)[0, :, :12, :2]
        mask = (
            data["gt_ego_fut_masks"]
            .unsqueeze(-1)
            .repeat(1, 1, 2)
            .unsqueeze(1)[0, :, :12, :2]
        )
        fut_boxes = data["fut_boxes"]
        row = {
            "index": index,
            "token": token,
            "scene_token": scene,
            "lt_eval_target": bool(info.get("lt_eval_target")),
            "lt_tpc_target": bool(info.get("lt_tpc_target")),
            "lt_event_types": list(info.get("lt_event_types", [])),
            "models": {},
        }

        for name, prediction_array in compact.items():
            pred_raw = torch.from_numpy(prediction_array[index]).to(dtype=torch.float32).unsqueeze(0)
            if pred_raw.shape != (1, 12, 2) or not torch.isfinite(pred_raw).all():
                raise RuntimeError(f"invalid compact prediction: {name}/{token}")
            model_row = {}

            if info.get("lt_eval_target"):
                if not bool(mask.all().item()):
                    raise RuntimeError(f"metric target has incomplete runtime mask: {token}")
                pred_eval = pred_raw.clone()
                gt_eval = gt.clone()
                pred_eval[..., 0] = -pred_eval[..., 0]
                gt_eval[..., 0] = -gt_eval[..., 0]
                l2 = metric_impl.compute_L2(pred_eval, gt_eval, mask).squeeze(0)
                _, box_col = metric_impl.evaluate_coll(
                    pred_eval[:, :, :2], gt_eval[:, :, :2], fut_boxes
                )
                l2_np = l2.detach().cpu().numpy().astype(np.float64)
                box_np = box_col.detach().cpu().numpy().astype(np.float64)
                accumulators[name]["L2"] += l2_np
                accumulators[name]["obj_box_col"] += box_np
                accumulators[name]["eval_count"] += 1
                scene_sums[name][scene]["L2"] += l2_np
                scene_sums[name][scene]["obj_box_col"] += box_np
                scene_sums[name][scene]["eval_count"] += 1
                model_row["L2_raw_per_step"] = [float(value) for value in l2_np.tolist()]
                model_row["BoxCol_raw_per_step"] = [float(value) for value in box_np.tolist()]

            if info.get("lt_tpc_target"):
                if previous[name] is None:
                    raise RuntimeError(f"TPC target lacks same-scene previous prediction: {name}/{token}")
                tpc = torch.sqrt((((pred_raw - previous[name]) ** 2) * mask).sum(dim=-1)).squeeze(0)
                tpc_np = tpc.detach().cpu().numpy().astype(np.float64)
                accumulators[name]["TPC"] += tpc_np
                accumulators[name]["tpc_count"] += 1
                scene_sums[name][scene]["TPC"] += tpc_np
                scene_sums[name][scene]["tpc_count"] += 1
                model_row["TPC_raw_per_step"] = [float(value) for value in tpc_np.tolist()]

            previous[name] = pred_raw.clone()
            if model_row:
                row["models"][name] = model_row

        if row["lt_eval_target"]:
            per_record.append(row)
        processed += 1

    if processed != EXPECTED_LT_RECORDS:
        raise RuntimeError(f"dataloader processed count mismatch: {processed}")

    metrics = {}
    for name, sums in accumulators.items():
        if sums["eval_count"] != EXPECTED_EVAL_TARGETS:
            raise RuntimeError(f"eval count mismatch for {name}: {sums['eval_count']}")
        if sums["tpc_count"] != EXPECTED_TPC_TARGETS:
            raise RuntimeError(f"TPC count mismatch for {name}: {sums['tpc_count']}")
        metrics[name] = {
            "L2": prefix_horizons(sums["L2"], sums["eval_count"]),
            "BoxCol_fraction": prefix_horizons(
                sums["obj_box_col"], sums["eval_count"]
            ),
            "TPC": prefix_horizons(sums["TPC"], sums["tpc_count"]),
        }
        metrics[name]["BoxCol_percent"] = {
            **metrics[name]["BoxCol_fraction"],
            "raw_per_step": [
                100.0 * value for value in metrics[name]["BoxCol_fraction"]["raw_per_step"]
            ],
            "horizons": {
                key: 100.0 * value
                for key, value in metrics[name]["BoxCol_fraction"]["horizons"].items()
            },
            "avg_1_6_horizons": 100.0
            * metrics[name]["BoxCol_fraction"]["avg_1_6_horizons"],
            "avg_4_6_horizons": 100.0
            * metrics[name]["BoxCol_fraction"]["avg_4_6_horizons"],
        }

    per_scene = {}
    for name, scenes in scene_sums.items():
        per_scene[name] = {}
        for scene, sums in scenes.items():
            per_scene[name][scene] = {
                "L2": prefix_horizons(sums["L2"], sums["eval_count"]),
                "BoxCol_fraction": prefix_horizons(
                    sums["obj_box_col"], sums["eval_count"]
                ),
                "TPC": prefix_horizons(sums["TPC"], sums["tpc_count"]),
            }
    return metrics, per_record, per_scene


def render_csv(metrics: dict) -> str:
    fieldnames = ["Method"]
    fieldnames += [f"L2@{second}s" for second in range(1, 7)]
    fieldnames += [f"BoxCol@{second}s (%)" for second in range(1, 7)]
    fieldnames += [f"TPC@{second}s" for second in range(1, 7)]
    fieldnames += ["TPC Avg. (1-6s)", "TPC Avg. (4-6s)"]
    rows = []
    for name, values in metrics.items():
        row = {"Method": name}
        for second in range(1, 7):
            horizon = f"{second}s"
            row[f"L2@{second}s"] = values["L2"]["horizons"][horizon]
            row[f"BoxCol@{second}s (%)"] = values["BoxCol_percent"]["horizons"][horizon]
            row[f"TPC@{second}s"] = values["TPC"]["horizons"][horizon]
        row["TPC Avg. (1-6s)"] = values["TPC"]["avg_1_6_horizons"]
        row["TPC Avg. (4-6s)"] = values["TPC"]["avg_4_6_horizons"]
        rows.append(row)

    import io

    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def main() -> int:
    args = parse_args()
    started = time.time()
    open_loop_root = Path(args.open_loop_root).resolve()
    config_path = Path(args.config).resolve()
    source_path = Path(args.source_pkl).resolve()
    lt_path = Path(args.lt_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    variants = parse_variants(args.variant)

    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not open_loop_root.is_dir() or not config_path.is_file():
        raise RuntimeError("open-loop root or config is missing")

    annotation_audit = validate_annotations(source_path, lt_path)
    variant_audit = validate_variant_files(variants)
    validate_dataset_runtime_order(
        open_loop_root,
        config_path,
        lt_path,
        annotation_audit["lt_infos"],
    )
    preflight = {
        "status": "PASS",
        "mode": "preflight-only" if args.preflight_only else "formal-offline-evaluation",
        "evaluator": str(Path(__file__).resolve()),
        "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "selection_uses_model_outputs": False,
        "new_training_started": False,
        "new_inference_started": False,
        "source_annotation": str(source_path),
        "source_annotation_sha256": annotation_audit["source_sha256"],
        "lt_annotation": str(lt_path),
        "lt_annotation_sha256": annotation_audit["lt_sha256"],
        "config": str(config_path),
        "open_loop_root": str(open_loop_root),
        "counts": {
            "source_records": EXPECTED_SOURCE_RECORDS,
            "lt_inference_context_records": EXPECTED_LT_RECORDS,
            "lt_scenes": EXPECTED_LT_SCENES,
            "l2_boxcol_targets": EXPECTED_EVAL_TARGETS,
            "tpc_targets": EXPECTED_TPC_TARGETS,
        },
        "runtime_order": annotation_audit["runtime_order"],
        "source_file_order_differs_from_runtime_order": annotation_audit[
            "source_file_order_differs_from_runtime_order"
        ],
        "lt_file_order_differs_from_runtime_order": annotation_audit[
            "lt_file_order_differs_from_runtime_order"
        ],
        "variants": [
            {
                key: str(value) if isinstance(value, Path) else value
                for key, value in item.items()
            }
            for item in variant_audit
        ],
    }
    atomic_write_json(out_dir / "preflight_audit.json", preflight)
    if args.preflight_only:
        print(json.dumps({"status": "PASS", "mode": "preflight-only"}))
        return 0

    compact, result_audit = compact_predictions(
        variant_audit,
        annotation_audit["selected_source_indices"],
        annotation_audit["lt_infos"],
        out_dir,
    )
    metrics, per_record, per_scene = evaluate_metrics(
        open_loop_root,
        config_path,
        lt_path,
        annotation_audit["lt_infos"],
        compact,
        args.workers,
    )

    output = {
        "status": "PASS",
        "protocol": {
            "inference_scope": "pre-existing full validation predictions; 642 complete-scene records retained as LT context",
            "l2_boxcol_scope": "lt_eval_target=true (189 records)",
            "tpc_scope": "lt_tpc_target=true (176 records)",
            "tpc_previous_state": "immediately preceding record in the same selected scene, including context-only records",
            "state_reset": "scene boundary",
            "selection_uses_model_outputs": False,
            "new_training_started": False,
            "new_inference_started": False,
        },
        "counts": preflight["counts"],
        "metrics": metrics,
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out_dir / "lt_nuscenes_metrics.json", output)
    atomic_write_json(out_dir / "per_record_metrics.json", per_record)
    atomic_write_json(out_dir / "per_scene_metrics.json", per_scene)
    atomic_write_json(out_dir / "source_results_audit.json", result_audit)
    atomic_write_text(out_dir / "lt_nuscenes_table.csv", render_csv(metrics))

    integrity = {}
    for path in sorted(out_dir.iterdir()):
        if path.is_file():
            integrity[path.name] = {
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
    atomic_write_json(out_dir / "output_integrity.json", integrity)
    print(
        json.dumps(
            {
                "status": "PASS",
                "out_dir": str(out_dir),
                "elapsed_seconds": output["elapsed_seconds"],
            }
        )
    )
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
