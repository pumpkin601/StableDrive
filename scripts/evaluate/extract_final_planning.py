#!/usr/bin/env python3
"""Memory-bounded extraction of final_planning from legacy results.pkl files."""
from __future__ import annotations

import argparse
import io
import pickle
import time
from pathlib import Path

import numpy as np
import torch


def keep_first_sample_memo(value, root):
    if value is root or isinstance(value, (str, int, float, bool, type(None), np.dtype)):
        return True
    if callable(value) or isinstance(value, type):
        return True
    if isinstance(value, bytes):
        return len(value) < 128
    if isinstance(value, tuple):
        scalar_types = (str, int, float, bool, type(None), type, np.dtype)
        return len(value) < 8 and all(isinstance(x, scalar_types) or callable(x) for x in value)
    return False


class PlanningOnlyUnpickler(pickle._Unpickler):
    """Discard detection tensors and per-sample pickle memo entries."""

    def find_class(self, module, name):
        if module == "torch.storage" and name == "_load_from_bytes":
            # final_planning's serialized storage is 339 bytes in these artifacts.
            return lambda blob: torch.load(io.BytesIO(blob), map_location="cpu") if len(blob) <= 400 else None
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            def rebuild(storage, offset, size, stride, requires_grad, hooks, metadata=None):
                if storage is None:
                    return None
                return torch._utils._rebuild_tensor_v2(
                    storage, offset, size, stride, requires_grad, hooks, metadata
                )
            return rebuild
        return super().find_class(module, name)

    def load_setitem(self):
        stack = self.stack
        value = stack.pop()
        key = stack.pop()
        result = stack[-1]
        result[key] = value
        if len(self.metastack) != 1 or key != "img_bbox":
            return

        plan = value.get("final_planning")
        result.clear()
        result["final_planning"] = plan

        root = self.metastack[0][0]
        memo_keys = list(self.memo)
        high_watermark = max(memo_keys)
        if not hasattr(self, "base_memo_high"):
            for memo_key in memo_keys:
                if not keep_first_sample_memo(self.memo[memo_key], root):
                    del self.memo[memo_key]
            self.base_memo_high = high_watermark
        else:
            for memo_key in memo_keys:
                if memo_key > self.base_memo_high:
                    del self.memo[memo_key]


PlanningOnlyUnpickler.dispatch[ord("s")] = PlanningOnlyUnpickler.load_setitem


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    start = time.time()
    with open(args.results, "rb") as f:
        rows = PlanningOnlyUnpickler(f).load()
    plans = []
    for index, row in enumerate(rows):
        plan = row.get("final_planning")
        if not isinstance(plan, torch.Tensor) or tuple(plan.shape) != (12, 2):
            raise RuntimeError(f"invalid final_planning at index {index}: {type(plan)} {getattr(plan, 'shape', None)}")
        plans.append(plan.detach().cpu().numpy().astype(np.float32))
    array = np.stack(plans)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, final_planning=array)
    print(f"saved {array.shape} to {out} in {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
