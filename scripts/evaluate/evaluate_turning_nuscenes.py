#!/usr/bin/env python3
"""Fail-closed evaluation on MomAD's official Turning-nuScenes subset.

The official release contains the 680 Turning-nuScenes tokens and 3-second
annotations.  StableDrive predicts 6 seconds, so this script maps those exact
tokens (without re-selecting any scene) to the frozen 6-second validation
annotation.  Existing full-validation predictions are then compacted by token
and evaluated for exactly two variants: Base Planner (A0) and StableDrive.

Two TPC protocols are reported:

* ``paper_compatible`` reproduces MomAD's released evaluator: incomplete
  targets are skipped, previous planning is initialized to zero, is updated
  only on complete targets, and is not reset between scenes;
* ``scene_reset_audit`` resets state at scene boundaries and does not score the
  first complete record of a scene.  It is an audit result, not a replacement
  for the paper-compatible comparison.

No training, inference, checkpoint selection, or output-dependent subset
selection is performed here.  All source artifacts are opened read-only and
verified by SHA-256 before evaluation.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch


EXPECTED_SOURCE_RECORDS = 6019
EXPECTED_TURNING_RECORDS = 680
EXPECTED_TURNING_SCENES = 17
EXPECTED_SOURCE_SHA256 = "c77120d97b683b7285e4a3440db52e7d806f054464f6747cf31b99f90bc6653a"
EXPECTED_OFFICIAL_TURNING_SHA256 = "f7f82e434c678321db4dbac1ec2e8875d24cbbcecdf6ebfa8f4a89a1bf7e8e65"
MODEL_ORDER = ["Base Planner (A0)", "StableDrive"]
HORIZON_STEPS = {f"{second}s": second * 2 for second in range(1, 7)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--open-loop-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-6s-pkl", required=True)
    parser.add_argument("--official-turning-pkl", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--variant",
        action="append",
        required=True,
        help=(
            "NAME|RESULTS_PKL|CHECKPOINT|EXPECTED_CHECKPOINT_SHA256|"
            "EXPECTED_RESULTS_SHA256"
        ),
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
    def find_class(self, module: str, name: str):
        if module == "torch.storage" and name == "_load_from_bytes":
            return lambda data: torch.load(io.BytesIO(data), map_location="cpu")
        return super().find_class(module, name)


def read_results_on_cpu(path: Path):
    with path.open("rb") as handle:
        return CPUUnpickler(handle).load()


def payload_infos(payload, path: Path) -> list:
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


def atomic_write_pickle(path: Path, payload) -> None:
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        raise RuntimeError(f"stale temporary output exists: {tmp}")
    with tmp.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def atomic_savez(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp.npz")
    if tmp.exists():
        raise RuntimeError(f"stale temporary output exists: {tmp}")
    np.savez_compressed(str(tmp), **arrays)
    with tmp.open("rb") as handle:
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def parse_variants(raw_specs: list[str]) -> list[dict]:
    variants = []
    seen = set()
    for raw in raw_specs:
        parts = raw.split("|", 4)
        if len(parts) != 5:
            raise RuntimeError(f"invalid --variant value: {raw}")
        name, results, checkpoint, checkpoint_sha, results_sha = parts
        if not name or name in seen:
            raise RuntimeError(f"empty or duplicate variant name: {name!r}")
        seen.add(name)
        variants.append(
            {
                "name": name,
                "results": Path(results).resolve(),
                "checkpoint": Path(checkpoint).resolve(),
                "expected_checkpoint_sha256": checkpoint_sha,
                "expected_results_sha256": results_sha,
            }
        )
    if [item["name"] for item in variants] != MODEL_ORDER:
        raise RuntimeError(f"variant identity/order mismatch: {[item['name'] for item in variants]}")
    return variants


def mask_for(info: dict, steps: int) -> np.ndarray:
    mask = np.asarray(info["gt_ego_fut_masks"]).reshape(-1)
    if mask.size < steps:
        raise RuntimeError(f"future mask shorter than {steps}: {info['token']}")
    return mask[:steps].astype(bool)


def validate_and_derive_annotations(
    source_path: Path,
    official_path: Path,
    derived_path: Path,
) -> dict:
    source_sha = sha256_file(source_path)
    official_sha = sha256_file(official_path)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise RuntimeError(f"source 6s annotation SHA mismatch: {source_sha}")
    if official_sha != EXPECTED_OFFICIAL_TURNING_SHA256:
        raise RuntimeError(f"official Turning annotation SHA mismatch: {official_sha}")

    source_payload = read_pickle(source_path)
    official_payload = read_pickle(official_path)
    source_raw = payload_infos(source_payload, source_path)
    official_raw = payload_infos(official_payload, official_path)
    if len(source_raw) != EXPECTED_SOURCE_RECORDS:
        raise RuntimeError(f"source record count mismatch: {len(source_raw)}")
    if len(official_raw) != EXPECTED_TURNING_RECORDS:
        raise RuntimeError(f"official Turning record count mismatch: {len(official_raw)}")

    source_tokens_raw = [str(info["token"]) for info in source_raw]
    official_tokens_raw = [str(info["token"]) for info in official_raw]
    if len(set(source_tokens_raw)) != len(source_tokens_raw):
        raise RuntimeError("duplicate token in source annotation")
    if len(set(official_tokens_raw)) != len(official_tokens_raw):
        raise RuntimeError("duplicate token in official Turning annotation")
    source_by_token = dict(zip(source_tokens_raw, source_raw))
    missing = sorted(set(official_tokens_raw) - set(source_tokens_raw))
    if missing:
        raise RuntimeError(f"official Turning tokens absent from 6s source: {len(missing)}")

    official_scenes = {str(info["scene_token"]) for info in official_raw}
    if len(official_scenes) != EXPECTED_TURNING_SCENES:
        raise RuntimeError(f"official Turning scene count mismatch: {len(official_scenes)}")
    source_scene_tokens = {
        str(info["token"])
        for info in source_raw
        if str(info["scene_token"]) in official_scenes
    }
    if source_scene_tokens != set(official_tokens_raw):
        raise RuntimeError(
            "official token set does not equal the complete records of its 17 source scenes: "
            f"official={len(set(official_tokens_raw))}, source-scenes={len(source_scene_tokens)}"
        )

    # Preserve the exact official raw token order. The dataset loader later
    # applies its standard ascending-timestamp runtime ordering.
    derived_raw = [source_by_token[token] for token in official_tokens_raw]
    for official_info, source_info in zip(official_raw, derived_raw):
        token = str(official_info["token"])
        if str(source_info["scene_token"]) != str(official_info["scene_token"]):
            raise RuntimeError(f"scene mismatch while mapping official token: {token}")
        official_traj = np.asarray(official_info["gt_ego_fut_trajs"])
        source_traj = np.asarray(source_info["gt_ego_fut_trajs"])
        if official_traj.shape != (6, 2) or source_traj.shape[0] < 12:
            raise RuntimeError(
                f"unexpected GT shape for {token}: official={official_traj.shape}, "
                f"source={source_traj.shape}"
            )
        if not np.allclose(official_traj, source_traj[:6, :2], rtol=0.0, atol=1e-6):
            raise RuntimeError(f"first-3s trajectory mismatch for official token: {token}")
        official_mask = mask_for(official_info, 6)
        source_mask = mask_for(source_info, 6)
        if not np.array_equal(official_mask, source_mask):
            raise RuntimeError(f"first-3s mask mismatch for official token: {token}")

    if not isinstance(source_payload, dict):
        raise RuntimeError("6s source annotation must contain metadata")
    derived_payload = {
        key: value for key, value in source_payload.items() if key != "infos"
    }
    derived_payload["infos"] = derived_raw
    atomic_write_pickle(derived_path, derived_payload)

    source_runtime = sorted(source_raw, key=lambda info: info["timestamp"])
    official_runtime = sorted(official_raw, key=lambda info: info["timestamp"])
    derived_runtime = sorted(derived_raw, key=lambda info: info["timestamp"])
    official_runtime_tokens = [str(info["token"]) for info in official_runtime]
    derived_runtime_tokens = [str(info["token"]) for info in derived_runtime]
    if official_runtime_tokens != derived_runtime_tokens:
        raise RuntimeError("3s official and derived 6s runtime token orders differ")
    source_runtime_tokens = [str(info["token"]) for info in source_runtime]
    source_runtime_index = {token: index for index, token in enumerate(source_runtime_tokens)}
    selected_source_indices = [source_runtime_index[token] for token in derived_runtime_tokens]
    if selected_source_indices != sorted(selected_source_indices):
        raise RuntimeError("Turning runtime order does not preserve full validation runtime order")

    previous_scene = None
    seen_scenes = set()
    for info in derived_runtime:
        scene = str(info["scene_token"])
        if scene != previous_scene:
            if scene in seen_scenes:
                raise RuntimeError(f"scene is not contiguous in runtime order: {scene}")
            seen_scenes.add(scene)
            previous_scene = scene

    complete_3s = sum(mask_for(info, 6).all() for info in derived_runtime)
    complete_6s = sum(mask_for(info, 12).all() for info in derived_runtime)
    if complete_3s <= complete_6s or complete_6s <= 0:
        raise RuntimeError(
            f"implausible complete-horizon counts: 3s={complete_3s}, 6s={complete_6s}"
        )

    return {
        "source_sha256": source_sha,
        "official_sha256": official_sha,
        "derived_sha256": sha256_file(derived_path),
        "source_runtime": source_runtime,
        "official_runtime": official_runtime,
        "derived_runtime": derived_runtime,
        "selected_source_indices": selected_source_indices,
        "complete_3s": int(complete_3s),
        "complete_6s": int(complete_6s),
        "official_raw_order_differs_from_runtime": official_tokens_raw != official_runtime_tokens,
        "runtime_order": "ascending timestamp, matching NuScenes3DDataset.load_annotations",
    }


def validate_variant_files(variants: list[dict]) -> list[dict]:
    audited = []
    for variant in variants:
        results = variant["results"]
        checkpoint = variant["checkpoint"]
        if not results.is_file() or not checkpoint.is_file():
            raise RuntimeError(f"missing artifact for {variant['name']}")
        checkpoint_sha = sha256_file(checkpoint)
        results_sha = sha256_file(results)
        if checkpoint_sha != variant["expected_checkpoint_sha256"]:
            raise RuntimeError(f"checkpoint SHA mismatch for {variant['name']}: {checkpoint_sha}")
        if results_sha != variant["expected_results_sha256"]:
            raise RuntimeError(f"results SHA mismatch for {variant['name']}: {results_sha}")
        audited.append(
            {
                **variant,
                "checkpoint_sha256": checkpoint_sha,
                "checkpoint_size_bytes": checkpoint.stat().st_size,
                "results_sha256": results_sha,
                "results_size_bytes": results.stat().st_size,
                "results_mtime_ns_before": results.stat().st_mtime_ns,
            }
        )
    return audited


def validate_dataset_runtime_order(
    open_loop_root: Path,
    config_path: Path,
    derived_path: Path,
    derived_runtime: list[dict],
) -> None:
    sys.path.insert(0, str(open_loop_root))
    os.chdir(str(open_loop_root))
    from mmcv import Config
    from mmdet.datasets import build_dataset

    import projects.mmdet3d_plugin  # noqa: F401

    cfg = Config.fromfile(str(config_path))
    eval_config = cfg.eval_config.copy()
    eval_config["ann_file"] = str(derived_path)
    dataset = build_dataset(eval_config)
    if len(dataset) != EXPECTED_TURNING_RECORDS:
        raise RuntimeError(f"built Turning dataset length mismatch: {len(dataset)}")
    actual = [str(info["token"]) for info in dataset.data_infos]
    expected = [str(info["token"]) for info in derived_runtime]
    if actual != expected:
        raise RuntimeError("built dataset order differs from audited official runtime order")


def compact_predictions(
    variants: list[dict],
    selected_source_indices: list[int],
    derived_runtime: list[dict],
    out_dir: Path,
) -> tuple[dict, list]:
    tokens = np.asarray([str(info["token"]) for info in derived_runtime], dtype="S32")
    scenes = np.asarray([str(info["scene_token"]) for info in derived_runtime], dtype="S32")
    compact = {}
    audit_rows = []
    for variant in variants:
        started = time.time()
        path = variant["results"]
        before = path.stat()
        results = read_results_on_cpu(path)
        if not isinstance(results, list) or len(results) != EXPECTED_SOURCE_RECORDS:
            raise RuntimeError(
                f"result length/type mismatch for {variant['name']}: "
                f"type={type(results).__name__}, len={len(results)}"
            )
        planning = np.empty((EXPECTED_TURNING_RECORDS, 12, 2), dtype=np.float32)
        for target_index, source_index in enumerate(selected_source_indices):
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
            planning[target_index] = value
        del results
        gc.collect()
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"source result changed during read: {variant['name']}")
        safe_name = "base_planner_a0" if variant["name"] == MODEL_ORDER[0] else "stabledrive"
        compact_path = out_dir / f"{safe_name}_turning_predictions.npz"
        atomic_savez(
            compact_path,
            token=tokens,
            scene_token=scenes,
            final_planning=planning,
        )
        compact[variant["name"]] = planning
        audit_rows.append(
            {
                "name": variant["name"],
                "results": str(path),
                "results_sha256": variant["results_sha256"],
                "results_size_bytes": before.st_size,
                "results_mtime_ns": before.st_mtime_ns,
                "compact_predictions": str(compact_path),
                "compact_predictions_sha256": sha256_file(compact_path),
                "compact_prediction_shape": list(planning.shape),
                "elapsed_seconds": time.time() - started,
            }
        )
    return compact, audit_rows


def metric_accumulator(steps: int) -> dict:
    return {
        "L2": np.zeros(steps, dtype=np.float64),
        "BoxCol": np.zeros(steps, dtype=np.float64),
        "TPC": np.zeros(steps, dtype=np.float64),
        "metric_count": 0,
        "tpc_count": 0,
    }


def summarize(raw_sum: np.ndarray, count: int, horizons: list[str]) -> dict:
    if count <= 0:
        raise RuntimeError("zero metric denominator")
    raw = np.asarray(raw_sum, dtype=np.float64) / float(count)
    values = {
        horizon: float(raw[: HORIZON_STEPS[horizon]].mean())
        for horizon in horizons
    }
    return {
        "records": int(count),
        "raw_per_step": raw.tolist(),
        "horizons": values,
        "horizon_average": float(np.mean(list(values.values()))),
    }


def evaluate_metrics(
    open_loop_root: Path,
    config_path: Path,
    derived_path: Path,
    derived_runtime: list[dict],
    compact: dict[str, np.ndarray],
    workers: int,
) -> tuple[dict, list, dict]:
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
    eval_config["ann_file"] = str(derived_path)
    dataset = build_dataset(eval_config)
    if [str(info["token"]) for info in dataset.data_infos] != [
        str(info["token"]) for info in derived_runtime
    ]:
        raise RuntimeError("evaluation dataset order mismatch")
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=max(0, int(workers)),
        shuffle=False,
        dist=False,
    )
    metric_impl = PlanningMetric()
    scopes = {
        "3s": {"steps": 6, "horizons": ["1s", "2s", "3s"]},
        "6s": {"steps": 12, "horizons": ["1s", "2s", "3s", "4s", "5s", "6s"]},
    }
    accum = {
        protocol: {
            scope: {name: metric_accumulator(spec["steps"]) for name in MODEL_ORDER}
            for scope, spec in scopes.items()
        }
        for protocol in ["paper_compatible", "scene_reset_audit"]
    }
    paper_previous = {
        scope: {
            name: torch.zeros((1, spec["steps"], 2), dtype=torch.float32)
            for name in MODEL_ORDER
        }
        for scope, spec in scopes.items()
    }
    audit_previous = {
        scope: {name: None for name in MODEL_ORDER} for scope in scopes
    }
    previous_scene = None
    per_record = []
    per_scene = {
        protocol: {
            scope: {
                name: defaultdict(lambda steps=spec["steps"]: metric_accumulator(steps))
                for name in MODEL_ORDER
            }
            for scope, spec in scopes.items()
        }
        for protocol in ["paper_compatible", "scene_reset_audit"]
    }

    for index, data in enumerate(dataloader):
        info = derived_runtime[index]
        token = str(info["token"])
        scene = str(info["scene_token"])
        batch_token = str(data["img_metas"].data[0][0]["token"])
        if batch_token != token:
            raise RuntimeError(f"dataloader token mismatch at {index}: {batch_token} != {token}")
        if scene != previous_scene:
            audit_previous = {
                scope: {name: None for name in MODEL_ORDER} for scope in scopes
            }
            previous_scene = scene

        gt_full = data["gt_ego_fut_trajs"].cumsum(dim=-2).unsqueeze(1)[0, :, :12, :2]
        mask_full = (
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
            "complete_3s": bool(mask_full[:, :6].all().item()),
            "complete_6s": bool(mask_full[:, :12].all().item()),
            "models": {},
        }
        for name in MODEL_ORDER:
            pred_full = torch.from_numpy(compact[name][index]).to(torch.float32).unsqueeze(0)
            if pred_full.shape != (1, 12, 2) or not torch.isfinite(pred_full).all():
                raise RuntimeError(f"invalid compact prediction: {name}/{token}")
            model_row = {}
            for scope, spec in scopes.items():
                steps = spec["steps"]
                complete = row[f"complete_{scope}"]
                if not complete:
                    continue
                pred_raw = pred_full[:, :steps].clone()
                gt = gt_full[:, :steps].clone()
                mask = mask_full[:, :steps].clone()
                pred_eval = pred_raw.clone()
                gt_eval = gt.clone()
                pred_eval[..., 0] = -pred_eval[..., 0]
                gt_eval[..., 0] = -gt_eval[..., 0]
                l2 = metric_impl.compute_L2(pred_eval, gt_eval, mask).squeeze(0)
                _, box = metric_impl.evaluate_coll(
                    pred_eval[:, :, :2], gt_eval[:, :, :2], fut_boxes
                )
                l2_np = l2.detach().cpu().numpy().astype(np.float64)
                box_np = box[:steps].detach().cpu().numpy().astype(np.float64)
                if l2_np.shape != (steps,) or box_np.shape != (steps,):
                    raise RuntimeError(f"metric shape mismatch: {name}/{token}/{scope}")

                paper_tpc = torch.sqrt(
                    (((pred_raw - paper_previous[scope][name]) ** 2) * mask).sum(dim=-1)
                ).squeeze(0)
                paper_tpc_np = paper_tpc.detach().cpu().numpy().astype(np.float64)
                paper_previous[scope][name] = pred_raw.clone()

                for protocol in ["paper_compatible", "scene_reset_audit"]:
                    target = accum[protocol][scope][name]
                    scene_target = per_scene[protocol][scope][name][scene]
                    target["L2"] += l2_np
                    target["BoxCol"] += box_np
                    target["metric_count"] += 1
                    scene_target["L2"] += l2_np
                    scene_target["BoxCol"] += box_np
                    scene_target["metric_count"] += 1

                paper_target = accum["paper_compatible"][scope][name]
                paper_scene = per_scene["paper_compatible"][scope][name][scene]
                paper_target["TPC"] += paper_tpc_np
                paper_target["tpc_count"] += 1
                paper_scene["TPC"] += paper_tpc_np
                paper_scene["tpc_count"] += 1

                audit_tpc_np = None
                if audit_previous[scope][name] is not None:
                    audit_tpc = torch.sqrt(
                        (((pred_raw - audit_previous[scope][name]) ** 2) * mask).sum(dim=-1)
                    ).squeeze(0)
                    audit_tpc_np = audit_tpc.detach().cpu().numpy().astype(np.float64)
                    audit_target = accum["scene_reset_audit"][scope][name]
                    audit_scene = per_scene["scene_reset_audit"][scope][name][scene]
                    audit_target["TPC"] += audit_tpc_np
                    audit_target["tpc_count"] += 1
                    audit_scene["TPC"] += audit_tpc_np
                    audit_scene["tpc_count"] += 1
                audit_previous[scope][name] = pred_raw.clone()

                model_row[scope] = {
                    "L2_raw_per_step": l2_np.tolist(),
                    "BoxCol_raw_per_step": box_np.tolist(),
                    "TPC_paper_compatible_raw_per_step": paper_tpc_np.tolist(),
                    "TPC_scene_reset_raw_per_step": (
                        audit_tpc_np.tolist() if audit_tpc_np is not None else None
                    ),
                }
            row["models"][name] = model_row
        per_record.append(row)

    if len(per_record) != EXPECTED_TURNING_RECORDS:
        raise RuntimeError(f"processed record count mismatch: {len(per_record)}")

    metrics = {}
    expected_complete = {
        "3s": sum(row["complete_3s"] for row in per_record),
        "6s": sum(row["complete_6s"] for row in per_record),
    }
    for protocol, scope_values in accum.items():
        metrics[protocol] = {}
        for scope, models in scope_values.items():
            metrics[protocol][scope] = {}
            for name, values in models.items():
                if values["metric_count"] != expected_complete[scope]:
                    raise RuntimeError(f"metric count mismatch: {protocol}/{scope}/{name}")
                expected_tpc = expected_complete[scope]
                if protocol == "scene_reset_audit":
                    expected_tpc -= EXPECTED_TURNING_SCENES
                if values["tpc_count"] != expected_tpc:
                    raise RuntimeError(
                        f"TPC count mismatch: {protocol}/{scope}/{name}: "
                        f"{values['tpc_count']} != {expected_tpc}"
                    )
                l2 = summarize(values["L2"], values["metric_count"], scopes[scope]["horizons"])
                box_fraction = summarize(
                    values["BoxCol"], values["metric_count"], scopes[scope]["horizons"]
                )
                tpc = summarize(values["TPC"], values["tpc_count"], scopes[scope]["horizons"])
                metrics[protocol][scope][name] = {
                    "L2": l2,
                    "BoxCol_fraction": box_fraction,
                    "BoxCol_percent": {
                        **box_fraction,
                        "raw_per_step": [100.0 * value for value in box_fraction["raw_per_step"]],
                        "horizons": {
                            key: 100.0 * value for key, value in box_fraction["horizons"].items()
                        },
                        "horizon_average": 100.0 * box_fraction["horizon_average"],
                    },
                    "TPC": tpc,
                }

    scene_output = {}
    for protocol, scope_values in per_scene.items():
        scene_output[protocol] = {}
        for scope, models in scope_values.items():
            scene_output[protocol][scope] = {}
            for name, scenes in models.items():
                scene_output[protocol][scope][name] = {}
                for scene, values in scenes.items():
                    item = {
                        "metric_count": values["metric_count"],
                        "tpc_count": values["tpc_count"],
                        "L2_sum_per_step": values["L2"].tolist(),
                        "BoxCol_sum_per_step": values["BoxCol"].tolist(),
                        "TPC_sum_per_step": values["TPC"].tolist(),
                    }
                    scene_output[protocol][scope][name][scene] = item
    return metrics, per_record, scene_output


def combined_values(metrics: dict, protocol: str, name: str, metric: str) -> dict:
    short = metrics[protocol]["3s"][name][metric]["horizons"]
    long = metrics[protocol]["6s"][name][metric]["horizons"]
    return {
        **{f"{second}s": short[f"{second}s"] for second in range(1, 4)},
        **{f"{second}s": long[f"{second}s"] for second in range(4, 7)},
    }


def render_csv(metrics: dict, protocol: str) -> str:
    fields = ["Method"]
    fields += [f"L2@{second}s" for second in range(1, 7)]
    fields += [f"BoxCol@{second}s (%)" for second in range(1, 7)]
    fields += [f"TPC@{second}s" for second in range(1, 7)]
    fields += ["TPC Avg. (1-6s)"]
    rows = []
    for name in MODEL_ORDER:
        row = {"Method": name}
        for metric, prefix in [
            ("L2", "L2"),
            ("BoxCol_percent", "BoxCol"),
            ("TPC", "TPC"),
        ]:
            values = combined_values(metrics, protocol, name, metric)
            for second in range(1, 7):
                row[f"{prefix}@{second}s" + (" (%)" if prefix == "BoxCol" else "")] = values[
                    f"{second}s"
                ]
            if metric == "TPC":
                row["TPC Avg. (1-6s)"] = float(np.mean(list(values.values())))
        rows.append(row)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue()


def render_summary(metrics: dict, counts: dict) -> str:
    lines = [
        "# Official Turning-nuScenes: A0 vs StableDrive",
        "",
        "- Subset: MomAD official Turning-nuScenes token set (17 scenes, 680 records).",
        f"- Complete 3 s targets: {counts['complete_3s']}.",
        f"- Complete 6 s targets: {counts['complete_6s']}.",
        "- Primary protocol: paper-compatible MomAD evaluator state handling.",
        "- Audit protocol: TPC state reset at every scene boundary.",
        "- No new training or inference was run; frozen full-validation predictions were extracted by token.",
        "",
        "## Paper-compatible results",
        "",
        "| Method | L2@4s | L2@5s | L2@6s | BoxCol@4s (%) | BoxCol@5s (%) | BoxCol@6s (%) | TPC@4s | TPC@5s | TPC@6s |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    protocol = "paper_compatible"
    for name in MODEL_ORDER:
        l2 = metrics[protocol]["6s"][name]["L2"]["horizons"]
        box = metrics[protocol]["6s"][name]["BoxCol_percent"]["horizons"]
        tpc = metrics[protocol]["6s"][name]["TPC"]["horizons"]
        lines.append(
            f"| {name} | {l2['4s']:.4f} | {l2['5s']:.4f} | {l2['6s']:.4f} | "
            f"{box['4s']:.4f} | {box['5s']:.4f} | {box['6s']:.4f} | "
            f"{tpc['4s']:.4f} | {tpc['5s']:.4f} | {tpc['6s']:.4f} |"
        )
    lines += [
        "",
        "Official MomAD paper reference for MomAD on Turning-nuScenes (4/5/6 s): "
        "L2 1.80/2.07/2.51, collision 0.85/1.57/2.31, TPC 1.37/1.58/1.93.",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    started = time.time()
    evaluator_path = Path(__file__).resolve()
    open_loop_root = Path(args.open_loop_root).resolve()
    config_path = Path(args.config).resolve()
    source_path = Path(args.source_6s_pkl).resolve()
    official_path = Path(args.official_turning_pkl).resolve()
    out_dir = Path(args.out_dir).resolve()
    variants = parse_variants(args.variant)
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(f"refusing to overwrite non-empty output directory: {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    if not open_loop_root.is_dir() or not config_path.is_file():
        raise RuntimeError("open-loop root or config is missing")

    derived_path = out_dir / "official_turning_nuscenes_6s_exact_tokens.pkl"
    annotation = validate_and_derive_annotations(source_path, official_path, derived_path)
    variant_audit = validate_variant_files(variants)
    validate_dataset_runtime_order(
        open_loop_root, config_path, derived_path, annotation["derived_runtime"]
    )
    counts = {
        "source_records": EXPECTED_SOURCE_RECORDS,
        "turning_records": EXPECTED_TURNING_RECORDS,
        "turning_scenes": EXPECTED_TURNING_SCENES,
        "complete_3s": annotation["complete_3s"],
        "complete_6s": annotation["complete_6s"],
    }
    preflight = {
        "status": "PASS",
        "mode": "preflight-only" if args.preflight_only else "formal-offline-evaluation",
        "evaluator": str(evaluator_path),
        "evaluator_sha256": sha256_file(evaluator_path),
        "official_subset_source": str(official_path),
        "official_subset_sha256": annotation["official_sha256"],
        "source_6s_annotation": str(source_path),
        "source_6s_annotation_sha256": annotation["source_sha256"],
        "derived_6s_annotation": str(derived_path),
        "derived_6s_annotation_sha256": annotation["derived_sha256"],
        "token_mapping": "exact official 680-token set; no new subset selection",
        "first_3s_gt_and_masks_match_official_release": True,
        "runtime_order": annotation["runtime_order"],
        "official_raw_order_differs_from_runtime": annotation[
            "official_raw_order_differs_from_runtime"
        ],
        "selection_uses_model_outputs": False,
        "new_training_started": False,
        "new_inference_started": False,
        "counts": counts,
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
        print(json.dumps({"status": "PASS", "mode": "preflight-only", "counts": counts}))
        return 0

    compact, result_audit = compact_predictions(
        variant_audit,
        annotation["selected_source_indices"],
        annotation["derived_runtime"],
        out_dir,
    )
    metrics, per_record, per_scene = evaluate_metrics(
        open_loop_root,
        config_path,
        derived_path,
        annotation["derived_runtime"],
        compact,
        args.workers,
    )
    output = {
        "status": "PASS",
        "protocol": {
            "subset": "MomAD official Turning-nuScenes 17-scene/680-token release",
            "six_second_mapping": "exact-token join to frozen 6019-record 6s annotation",
            "three_second_scope": "first 6 valid future steps; GT and masks verified against official release",
            "six_second_scope": "all 12 future steps valid",
            "paper_compatible_tpc": "zero initialization; complete targets only; no scene reset",
            "scene_reset_audit_tpc": "complete targets only; reset at scene boundary; scene-first complete target unscored",
            "selection_uses_model_outputs": False,
            "new_training_started": False,
            "new_inference_started": False,
        },
        "counts": counts,
        "metrics": metrics,
        "official_momad_paper_reference_4_5_6s": {
            "L2": [1.80, 2.07, 2.51],
            "BoxCol_percent": [0.85, 1.57, 2.31],
            "TPC": [1.37, 1.58, 1.93],
        },
        "elapsed_seconds": time.time() - started,
    }
    atomic_write_json(out_dir / "official_turning_metrics.json", output)
    atomic_write_json(out_dir / "per_record_metrics.json", per_record)
    atomic_write_json(out_dir / "per_scene_sums.json", per_scene)
    atomic_write_json(out_dir / "source_results_audit.json", result_audit)
    atomic_write_text(
        out_dir / "paper_compatible_table.csv", render_csv(metrics, "paper_compatible")
    )
    atomic_write_text(
        out_dir / "scene_reset_audit_table.csv", render_csv(metrics, "scene_reset_audit")
    )
    atomic_write_text(out_dir / "REPORT.md", render_summary(metrics, counts))

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
                "counts": counts,
                "elapsed_seconds": output["elapsed_seconds"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
