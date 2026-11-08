#!/usr/bin/env python3
"""Fast extraction of ego-future GT from the protocol-2 info pickle.

The legacy annotation pickle memoizes ``token`` as key 6 and
``gt_ego_fut_trajs`` as key 374.  Scanning those opcodes avoids materializing
camera/sweep annotations and is therefore both faster and memory bounded.
"""
from __future__ import annotations

import argparse
import bisect
import json
import mmap
import struct
from pathlib import Path

import numpy as np


TOKEN_KEY = b"h\x06"                       # BINGET 6
GT_KEY = b"j\x76\x01\x00\x00"           # LONG_BINGET 374


def positions(blob, pattern):
    out, offset = [], 0
    while True:
        offset = blob.find(pattern, offset)
        if offset < 0:
            return out
        out.append(offset)
        offset += len(pattern)


def next_token(blob, start, allowed):
    end = min(start + 100, len(blob))
    pos = start
    while pos < end:
        pos = blob.find(b"X", pos, end)  # BINUNICODE
        if pos < 0:
            return None
        length = struct.unpack_from("<I", blob, pos + 1)[0]
        raw = blob[pos + 5:pos + 5 + length]
        if length == 32:
            try:
                token = raw.decode("ascii")
            except UnicodeDecodeError:
                token = ""
            if token in allowed:
                return token
        pos += 1
    return None


def next_gt(blob, start):
    end = min(start + 500, len(blob))
    pos = start
    while pos < end:
        pos = blob.find(b"X", pos, end)
        if pos < 0:
            return None
        length = struct.unpack_from("<I", blob, pos + 1)[0]
        if 80 <= length <= 300:
            raw = blob[pos + 5:pos + 5 + length]
            try:
                decoded = raw.decode("utf-8").encode("latin1")
            except UnicodeDecodeError:
                decoded = b""
            if len(decoded) == 12 * 2 * 4:
                return np.frombuffer(decoded, dtype="<f4").reshape(12, 2).copy()
        pos += 1
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infos", required=True)
    ap.add_argument("--subset-stats", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    stats = json.load(open(args.subset_stats, encoding="utf-8"))["samples"]
    allowed = set(stats)
    with open(args.infos, "rb") as f:
        blob = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        token_starts = [blob.find(b"token")] + positions(blob, TOKEN_KEY)
        gt_starts = [blob.find(b"gt_ego_fut_trajs")] + positions(blob, GT_KEY)
        token_events = []
        for start in token_starts:
            token = next_token(blob, start, allowed)
            if token is not None:
                token_events.append((start, token))
        trajectories = [next_gt(blob, start) for start in gt_starts]
        token_positions = [x[0] for x in token_events]
        token_order = [
            token_events[bisect.bisect_right(token_positions, gt_start) - 1][1]
            for gt_start in gt_starts
        ]

    if len(token_order) != 6019 or len(set(token_order)) != 6019 or len(trajectories) != 6019 or any(x is None for x in trajectories):
        raise RuntimeError(
            f"extraction mismatch: tokens={len(token_order)}, unique_tokens={len(set(token_order))}, "
            f"gt={len(trajectories)}, missing_gt={sum(x is None for x in trajectories)}"
        )
    relative = np.stack(trajectories).astype(np.float32)
    absolute = np.cumsum(relative, axis=1)
    full_valid = np.asarray([bool(stats[token]["full_valid_gt"]) for token in token_order])
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, token=np.asarray(token_order), gt=absolute, full_valid=full_valid)
    print(f"saved {len(token_order)} GT trajectories to {out}")


if __name__ == "__main__":
    main()
