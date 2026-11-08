#!/usr/bin/env python3
"""Build the deployable StableDrive checkpoint from T0 and T1-off.

Only shared state is interpolated. T1-only MSTS adapter keys are retired, and
non-floating buffers must be identical. No optimizer, backward pass, or
training step is created by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t0", type=Path, required=True)
    parser.add_argument("--t1-off", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--expected-retired-keys", type=int, default=42)
    parser.add_argument("--retired-key-marker", default="c3a_horizon_adapter")
    parser.add_argument("--iteration", type=int, default=11720)
    return parser.parse_args()


def checkpoint_state(payload: dict) -> dict:
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict) or not state:
        raise TypeError("checkpoint does not contain a non-empty state_dict")
    return state


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite {args.output}")

    t0 = checkpoint_state(torch.load(args.t0, map_location="cpu"))
    t1 = checkpoint_state(torch.load(args.t1_off, map_location="cpu"))
    missing = sorted(set(t0) - set(t1))
    if missing:
        raise RuntimeError(f"T1-off is missing {len(missing)} T0 keys")

    retired = sorted(set(t1) - set(t0))
    if any(args.retired_key_marker not in key for key in retired):
        raise RuntimeError("T1-only keys are not exclusively MSTS adapter keys")
    if args.expected_retired_keys >= 0 and len(retired) != args.expected_retired_keys:
        raise RuntimeError(
            f"expected {args.expected_retired_keys} retired keys, found {len(retired)}"
        )

    state = {}
    for key in sorted(t0):
        left, right = t0[key], t1[key]
        if left.shape != right.shape or left.dtype != right.dtype:
            raise RuntimeError(f"endpoint incompatibility at {key}")
        if torch.is_floating_point(left):
            value = left.mul(1.0 - args.alpha).add(right, alpha=args.alpha)
            if not torch.isfinite(value).all():
                raise RuntimeError(f"non-finite interpolation at {key}")
        else:
            if not torch.equal(left, right):
                raise RuntimeError(f"non-floating endpoint mismatch at {key}")
            value = left.clone()
        state[key] = value

    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": {
            "operation": "zero_training_shared_weight_linear_interpolation",
            "alpha": args.alpha,
            "T0_sha256": sha256_file(args.t0),
            "T1_off_sha256": sha256_file(args.t1_off),
            "shared_key_count": len(state),
            "retired_msts_key_count": len(retired),
            "optimizer_created": False,
            "backward_called": False,
            "training_started": False,
        },
        "state_dict": state,
        "iteration": args.iteration,
    }
    torch.save(payload, args.output)
    manifest = {
        **payload["meta"],
        "output": str(args.output.resolve()),
        "output_sha256": sha256_file(args.output),
        "retired_keys": retired,
    }
    manifest_path = args.output.with_suffix(args.output.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
