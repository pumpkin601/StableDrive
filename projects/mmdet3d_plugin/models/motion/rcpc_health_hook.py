from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import torch
import torch.distributed as dist
from mmcv.runner import HOOKS, Hook


@HOOKS.register_module()
class RCPCDistributedHealthHook(Hook):
    """Fail fast when RCPC parameters diverge or become non-finite under DDP."""

    def __init__(
        self,
        interval: int = 64,
        sample_count_per_parameter: int = 8,
        max_parameter_difference: float = 1e-6,
        output_name: str = "rcpc_ddp_health.json",
    ) -> None:
        self.interval = int(interval)
        self.sample_count_per_parameter = int(sample_count_per_parameter)
        self.max_parameter_difference = float(max_parameter_difference)
        self.output_name = str(output_name)
        self.records: List[Dict[str, object]] = []
        self._last_checked_update = 0
        if self.interval < 1:
            raise ValueError("RCPC health interval must be positive")
        if self.sample_count_per_parameter < 1:
            raise ValueError("RCPC health sample count must be positive")

    def _raw_model(self, runner):
        return runner.model.module if hasattr(runner.model, "module") else runner.model

    def _parameter_signature(self, parameter: torch.Tensor) -> torch.Tensor:
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
        statistics = torch.stack(
            [
                flat.mean(),
                flat.square().mean(),
                flat.abs().max(),
            ]
        )
        return torch.cat([sampled, statistics])

    def _check(self, runner, successful_update: int) -> None:
        if successful_update == self._last_checked_update:
            return
        model = self._raw_model(runner)
        parameters = [
            (name, parameter)
            for name, parameter in model.named_parameters()
            if ".rcpc_adapter." in name or name.startswith("rcpc_adapter.")
        ]
        if not parameters:
            raise RuntimeError("RCPC health hook found no RCPC parameters")
        signature = torch.cat(
            [self._parameter_signature(parameter) for _, parameter in parameters]
        )
        local_nonfinite = torch.tensor(
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
            dist.all_reduce(local_nonfinite, op=dist.ReduceOp.MAX)
            gathered = [torch.empty_like(signature) for _ in range(world_size)]
            dist.all_gather(gathered, signature)
            reference = gathered[0]
            max_difference = torch.stack(
                [(item - reference).abs().max() for item in gathered]
            ).max()
        grad_present = sum(
            int(parameter.grad is not None) for _, parameter in parameters
        )
        record = {
            "successful_update": int(successful_update),
            "rank": int(rank),
            "world_size": int(world_size),
            "rcpc_parameter_count": len(parameters),
            "rcpc_grad_present_parameter_count": int(grad_present),
            "nonfinite_detected": bool(local_nonfinite.item()),
            "max_rank_parameter_difference": float(max_difference.item()),
            "tolerance": self.max_parameter_difference,
            "status": "PASS",
        }
        if record["nonfinite_detected"]:
            record["status"] = "FAIL_NONFINITE"
        elif record["max_rank_parameter_difference"] > self.max_parameter_difference:
            record["status"] = "FAIL_DDP_DIVERGENCE"
        self.records.append(record)
        self._last_checked_update = int(successful_update)
        runner.log_buffer.output["rcpc_ddp_max_param_diff"] = record[
            "max_rank_parameter_difference"
        ]
        runner.log_buffer.output["rcpc_ddp_health_pass"] = float(
            record["status"] == "PASS"
        )
        if record["status"] != "PASS":
            raise RuntimeError("RCPC distributed health failure: %s" % record)

    def after_train_iter(self, runner) -> None:
        successful_update = int(runner.iter) + 1
        if successful_update % self.interval == 0:
            self._check(runner, successful_update)

    def after_run(self, runner) -> None:
        # IterBasedRunner increments ``iter`` after after_train_iter, so it is
        # already the completed-update count when after_run is called.
        successful_update = int(runner.iter)
        self._check(runner, successful_update)
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return
        output_path = Path(runner.work_dir) / self.output_name
        payload = {
            "status": (
                "PASS"
                if self.records and all(item["status"] == "PASS" for item in self.records)
                else "FAIL"
            ),
            "interval": self.interval,
            "max_parameter_difference_tolerance": self.max_parameter_difference,
            "records": self.records,
        }
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


@HOOKS.register_module()
class RCPCFrozenBaseHook(Hook):
    """Keep the frozen base in inference mode and fail on gradient leakage."""

    def __init__(
        self,
        output_name: str = "rcpc_frozen_base_health.json",
    ) -> None:
        self.output_name = str(output_name)
        self.checked_updates = 0
        self.trainable_parameter_count = 0
        self.frozen_parameter_count = 0
        self.trainable_numel = 0
        self.frozen_numel = 0
        self._frozen_modules = []

    def _raw_model(self, runner):
        return runner.model.module if hasattr(runner.model, "module") else runner.model

    def _rcpc(self, model):
        head = model.head.motion_plan_head
        rcpc = getattr(head, "rcpc_adapter", None)
        if rcpc is None:
            raise RuntimeError("RCPCFrozenBaseHook requires an active rcpc_adapter")
        return rcpc

    def before_run(self, runner) -> None:
        model = self._raw_model(runner)
        trainable = []
        frozen = []
        bad_trainable = []
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                trainable.append((name, parameter))
                if not (
                    name.startswith("head.motion_plan_head.rcpc_adapter.")
                    or ".head.motion_plan_head.rcpc_adapter." in name
                ):
                    bad_trainable.append(name)
            else:
                frozen.append((name, parameter))
        if bad_trainable:
            raise RuntimeError(
                "Frozen-base audit found non-RCPC trainable parameters: %s"
                % bad_trainable[:20]
            )
        if not trainable or not frozen:
            raise RuntimeError("Frozen-base audit requires both trainable and frozen parameters")
        self.trainable_parameter_count = len(trainable)
        self.frozen_parameter_count = len(frozen)
        self.trainable_numel = sum(parameter.numel() for _, parameter in trainable)
        self.frozen_numel = sum(parameter.numel() for _, parameter in frozen)
        self._rcpc(model)
        # Keep the detector and ancestors of RCPC in training mode so
        # SparseDrive.forward dispatches to forward_train. Only submodules
        # with no trainable descendants are forced to inference behavior.
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
            raise RuntimeError("Frozen-base audit found no fully frozen submodules")

    def before_train_iter(self, runner) -> None:
        # IterBasedRunner calls model.train() before every iteration. Preserve
        # the top-level forward_train dispatch while restoring inference mode
        # only for submodules with no trainable descendants.
        model = self._raw_model(runner)
        for module in self._frozen_modules:
            module.eval()
        self._rcpc(model).train()

    def after_train_iter(self, runner) -> None:
        model = self._raw_model(runner)
        leaked = [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        if leaked:
            raise RuntimeError("Gradient leaked into frozen parameters: %s" % leaked[:20])
        trainable_without_grad = [
            name
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and parameter.grad is None
        ]
        if trainable_without_grad:
            raise RuntimeError(
                "Trainable RCPC parameters lack gradients: %s"
                % trainable_without_grad[:20]
            )
        self.checked_updates += 1
        runner.log_buffer.output["rcpc_frozen_base_grad_leak"] = 0.0
        runner.log_buffer.output["rcpc_trainable_grad_coverage"] = 1.0

    def after_run(self, runner) -> None:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        if rank != 0:
            return
        payload = {
            "status": "PASS" if self.checked_updates > 0 else "FAIL",
            "checked_updates": self.checked_updates,
            "trainable_parameter_count": self.trainable_parameter_count,
            "frozen_parameter_count": self.frozen_parameter_count,
            "trainable_numel": self.trainable_numel,
            "frozen_numel": self.frozen_numel,
            "trainable_scope": "head.motion_plan_head.rcpc_adapter.*",
            "base_forward_mode": "frozen_submodules_eval_with_forward_train_dispatch",
            "rcpc_forward_mode": "train",
            "frozen_gradient_leak_count": 0,
        }
        output_path = Path(runner.work_dir) / self.output_name
        output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
