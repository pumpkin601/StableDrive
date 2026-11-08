"""Frozen-base and distributed numerical-health checks for CSPC training."""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class CSPCDistributedHealthHook(Hook):
    """Enforce CSPC-only training and fail fast on DDP/numerical faults."""

    _SCOPE = "head.motion_plan_head.selective_pareto_controller"

    def __init__(
        self,
        interval=64,
        sample_count_per_parameter=8,
        max_parameter_difference=1e-6,
        output_name="cspc_ddp_health.json",
    ):
        self.interval = int(interval)
        self.sample_count_per_parameter = int(sample_count_per_parameter)
        self.max_parameter_difference = float(max_parameter_difference)
        self.output_name = str(output_name)
        self.records = []
        self._last_checked_update = 0
        self._trainable = []
        self._frozen = []
        self._frozen_modules = []
        self.checked_updates = 0
        if self.interval < 1 or self.sample_count_per_parameter < 1:
            raise ValueError("CSPC health intervals/sample counts must be positive")

    @staticmethod
    def _raw_model(runner):
        return runner.model.module if hasattr(runner.model, "module") else runner.model

    @classmethod
    def _in_scope(cls, name):
        return name == cls._SCOPE or name.startswith(cls._SCOPE + ".")

    def _controller(self, model):
        controller = getattr(
            model.head.motion_plan_head,
            "selective_pareto_controller",
            None,
        )
        if controller is None:
            raise RuntimeError("CSPC health hook requires an active controller")
        return controller

    def before_run(self, runner):
        model = self._raw_model(runner)
        self._trainable = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        ]
        self._frozen = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad
        ]
        invalid = [name for name, _ in self._trainable if not self._in_scope(name)]
        if invalid:
            raise RuntimeError(
                "CSPC-only contract found out-of-scope trainable parameters: %s"
                % invalid[:20]
            )
        if not self._trainable or not self._frozen:
            raise RuntimeError("CSPC health hook requires trainable and frozen tensors")
        self._controller(model)
        # Keep the top-level detector and controller ancestors in training mode
        # for forward_train dispatch. Every subtree without a trainable CSPC
        # descendant runs in inference mode, preventing dropout/BN drift.
        self._frozen_modules = [
            module
            for name, module in model.named_modules()
            if name
            and not any(
                parameter.requires_grad
                for parameter in module.parameters(recurse=True)
            )
        ]
        if not self._frozen_modules:
            raise RuntimeError("CSPC health hook found no frozen submodules")

    def before_train_iter(self, runner):
        model = self._raw_model(runner)
        for module in self._frozen_modules:
            module.eval()
        self._controller(model).train()

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

    def _distributed_check(self, runner, successful_update):
        if successful_update == self._last_checked_update:
            return
        signature = torch.cat(
            [self._signature(parameter) for _, parameter in self._trainable]
        )
        nonfinite = torch.tensor(
            int(not torch.isfinite(signature).all().item()),
            device=signature.device,
            dtype=torch.int64,
        )
        rank = 0
        world_size = 1
        max_difference = signature.new_zeros(())
        if dist.is_available() and dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
            dist.all_reduce(nonfinite, op=dist.ReduceOp.MAX)
            gathered = [torch.empty_like(signature) for _ in range(world_size)]
            dist.all_gather(gathered, signature)
            max_difference = torch.stack(
                [(item - gathered[0]).abs().max() for item in gathered]
            ).max()

        missing_gradients = [
            name for name, parameter in self._trainable if parameter.grad is None
        ]
        nonfinite_gradients = [
            name
            for name, parameter in self._trainable
            if parameter.grad is not None
            and not torch.isfinite(parameter.grad).all().item()
        ]
        leaked_gradients = [
            name for name, parameter in self._frozen if parameter.grad is not None
        ]
        record = {
            "successful_update": int(successful_update),
            "rank": int(rank),
            "world_size": int(world_size),
            "trainable_parameter_tensor_count": len(self._trainable),
            "frozen_parameter_tensor_count": len(self._frozen),
            "gradient_present_count": len(self._trainable) - len(missing_gradients),
            "missing_gradient_count": len(missing_gradients),
            "nonfinite_gradient_count": len(nonfinite_gradients),
            "frozen_gradient_leak_count": len(leaked_gradients),
            "nonfinite_parameter_detected": bool(nonfinite.item()),
            "max_rank_parameter_difference": float(max_difference.item()),
            "tolerance": self.max_parameter_difference,
            "status": "PASS",
        }
        if record["nonfinite_parameter_detected"]:
            record["status"] = "FAIL_NONFINITE_PARAMETER"
        elif nonfinite_gradients:
            record["status"] = "FAIL_NONFINITE_GRADIENT"
        elif missing_gradients:
            record["status"] = "FAIL_MISSING_GRADIENT"
        elif leaked_gradients:
            record["status"] = "FAIL_FROZEN_GRADIENT_LEAK"
        elif record["max_rank_parameter_difference"] > self.max_parameter_difference:
            record["status"] = "FAIL_DDP_DIVERGENCE"
        self.records.append(record)
        self._last_checked_update = int(successful_update)
        runner.log_buffer.output["cspc_ddp_max_param_diff"] = record[
            "max_rank_parameter_difference"
        ]
        runner.log_buffer.output["cspc_ddp_health_pass"] = float(
            record["status"] == "PASS"
        )
        if record["status"] != "PASS":
            raise RuntimeError("CSPC distributed health failure: %s" % record)

    def after_train_iter(self, runner):
        successful_update = int(runner.iter) + 1
        # Cheap local checks are applied to every update; the synchronized
        # parameter signature is evaluated at the locked interval.
        leaked = [
            name for name, parameter in self._frozen if parameter.grad is not None
        ]
        missing = [
            name for name, parameter in self._trainable if parameter.grad is None
        ]
        nonfinite = [
            name
            for name, parameter in self._trainable
            if parameter.grad is not None
            and not torch.isfinite(parameter.grad).all().item()
        ]
        if leaked or missing or nonfinite:
            raise RuntimeError(
                "CSPC gradient contract failure: leaked=%s missing=%s nonfinite=%s"
                % (leaked[:10], missing[:10], nonfinite[:10])
            )
        self.checked_updates += 1
        if successful_update % self.interval == 0:
            self._distributed_check(runner, successful_update)

    def after_run(self, runner):
        self._distributed_check(runner, int(runner.iter))
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return
        payload = {
            "status": (
                "PASS"
                if self.checked_updates > 0
                and self.records
                and all(record["status"] == "PASS" for record in self.records)
                else "FAIL"
            ),
            "checked_updates": self.checked_updates,
            "interval": self.interval,
            "trainable_scope": self._SCOPE + ".*",
            "trainable_parameter_tensor_count": len(self._trainable),
            "frozen_parameter_tensor_count": len(self._frozen),
            "trainable_numel": sum(value.numel() for _, value in self._trainable),
            "frozen_numel": sum(value.numel() for _, value in self._frozen),
            "max_parameter_difference_tolerance": self.max_parameter_difference,
            "base_forward_mode": "fully_frozen_submodules_eval",
            "controller_forward_mode": "train",
            "records": self.records,
        }
        path = Path(runner.work_dir) / self.output_name
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
