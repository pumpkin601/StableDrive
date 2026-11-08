"""Distributed numerical-health checks for the joint C3+CDRS module."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class C3C4DistributedHealthHook(Hook):
    """Fail when joint-module parameters become non-finite or diverge by rank."""

    def __init__(
        self,
        interval=64,
        sample_count_per_parameter=8,
        max_parameter_difference=1e-6,
        output_name="c3c4_ddp_health.json",
    ):
        self.interval = int(interval)
        self.sample_count_per_parameter = int(sample_count_per_parameter)
        self.max_parameter_difference = float(max_parameter_difference)
        self.output_name = str(output_name)
        self.records = []
        self._last_checked_update = 0
        if self.interval < 1 or self.sample_count_per_parameter < 1:
            raise ValueError("C3+C4 health intervals/sample counts must be positive")

    def _raw_model(self, runner):
        return runner.model.module if hasattr(runner.model, "module") else runner.model

    def _signature(self, parameter):
        flat = parameter.detach().float().reshape(-1)
        if flat.numel() <= self.sample_count_per_parameter:
            sampled = flat
        else:
            index = torch.linspace(
                0,
                flat.numel() - 1,
                self.sample_count_per_parameter,
                device=flat.device,
            ).long()
            sampled = flat[index]
        if sampled.numel() < self.sample_count_per_parameter:
            sampled = torch.nn.functional.pad(
                sampled, (0, self.sample_count_per_parameter - sampled.numel())
            )
        return torch.cat(
            [sampled, torch.stack([flat.mean(), flat.square().mean(), flat.abs().max()])]
        )

    def _check(self, runner, successful_update):
        if successful_update == self._last_checked_update:
            return
        model = self._raw_model(runner)
        groups = {
            "c3": [
                (name, value)
                for name, value in model.named_parameters()
                if ".c3a_horizon_adapter." in name
            ],
            "cdrs": [
                (name, value)
                for name, value in model.named_parameters()
                if ".dense_risk_scorer." in name
            ],
        }
        if any(not values for values in groups.values()):
            raise RuntimeError("C3+C4 health hook requires both C3 and CDRS")
        parameters = groups["c3"] + groups["cdrs"]
        signature = torch.cat([self._signature(value) for _, value in parameters])
        nonfinite = torch.tensor(
            int(not torch.isfinite(signature).all().item()),
            device=signature.device,
            dtype=torch.int64,
        )
        world_size = 1
        rank = 0
        max_difference = signature.new_zeros(())
        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            rank = dist.get_rank()
            dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX)
            gathered = [torch.empty_like(signature) for _ in range(world_size)]
            dist.all_gather(gathered, signature)
            max_difference = torch.stack(
                [(item - gathered[0]).abs().max() for item in gathered]
            ).max()
        grad_counts = {
            name: sum(int(value.grad is not None) for _, value in values)
            for name, values in groups.items()
        }
        record = {
            "successful_update": int(successful_update),
            "rank": int(rank),
            "world_size": int(world_size),
            "parameter_tensor_counts": {
                name: len(values) for name, values in groups.items()
            },
            "gradient_present_counts": grad_counts,
            "nonfinite_detected": bool(nonfinite.item()),
            "max_rank_parameter_difference": float(max_difference.item()),
            "tolerance": self.max_parameter_difference,
            "status": "PASS",
        }
        if record["nonfinite_detected"]:
            record["status"] = "FAIL_NONFINITE"
        elif record["max_rank_parameter_difference"] > self.max_parameter_difference:
            record["status"] = "FAIL_DDP_DIVERGENCE"
        elif any(
            grad_counts[name] != len(groups[name]) for name in ("c3", "cdrs")
        ):
            record["status"] = "FAIL_MISSING_GRADIENT"
        self.records.append(record)
        self._last_checked_update = int(successful_update)
        if record["status"] != "PASS":
            raise RuntimeError("C3+C4 distributed health failure: %s" % record)

    def after_train_iter(self, runner):
        successful_update = int(runner.iter) + 1
        if successful_update % self.interval == 0:
            self._check(runner, successful_update)

    def after_run(self, runner):
        self._check(runner, int(runner.iter))
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return
        payload = {
            "status": (
                "PASS"
                if self.records
                and all(record["status"] == "PASS" for record in self.records)
                else "FAIL"
            ),
            "interval": self.interval,
            "max_parameter_difference_tolerance": self.max_parameter_difference,
            "records": self.records,
        }
        path = Path(runner.work_dir) / self.output_name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
