from typing import List, Optional, Tuple, Union
import warnings
import copy
import os

import numpy as np
import cv2
import torch
import torch.nn as nn

from mmcv.utils import build_from_cfg
from mmcv.cnn import Linear, bias_init_with_prob
from mmcv.cnn.bricks.transformer import BaseTransformerLayer
from mmcv.runner import BaseModule, force_fp32
from mmcv.cnn.bricks.registry import (
    ATTENTION,
    PLUGIN_LAYERS,
    POSITIONAL_ENCODING,
    FEEDFORWARD_NETWORK,
    NORM_LAYERS,
)
from mmdet.core import reduce_mean
from mmdet.models import HEADS
from mmdet.core.bbox.builder import BBOX_SAMPLERS, BBOX_CODERS
from mmdet.models import build_loss
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners
from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.motion.motion_blocks import *

from ..attention import gen_sineembed_for_position
from ..blocks import linear_relu_ln
from ..instance_bank import topk
from nuscenes.nuscenes import NuScenes
from .causal_smm_adapter import CausalSMMAdapter
from .causal_alignment import (
    EgoPose2D,
    observed_velocity_xy_from_ego_status,
    quaternion_yaw,
    retrospective_calibration_features,
)
from .causal_history_sequence import (
    CausalHistoryCache,
    CausalHistoryEntry,
    CausalHistorySequenceRetriever,
)
from .longitudinal_phase_horizon_mamba import (
    C3AHorizonPlanningAdapter,
    phase_supervision_loss,
)
from .next_token_prediction import NextTokenPredictor
from .retrospective_pareto_critic import RCPCPlanningAdapter
from .causal_dense_risk_scorer import CDRSPlanningAdapter
from .causal_selective_pareto_controller import CSPCPlanningAdapter
from .smm_observability import SMMObservabilityObserver
@HEADS.register_module()
class MotionPlanningHeadroboAD_6s(BaseModule):
    def __init__(
        self,
        fut_ts=12,
        fut_mode=6,
        ego_fut_ts=12,
        ego_fut_mode=3,
        motion_anchor=None,
        plan_anchor=None,
        embed_dims=256,
        decouple_attn=False,
        instance_queue=None,
        operation_order=None,
        temp_graph_model=None,
        graph_model=None,
        cross_graph_model=None,
        norm_layer=None,
        ffn=None,
        refine_layer=None,
        motion_sampler=None,
        motion_loss_cls=None,
        motion_loss_reg=None,
        planning_sampler=None,
        plan_loss_cls=None,
        plan_loss_reg=None,
        plan_loss_status=None,
        motion_decoder=None,
        planning_decoder=None,
        num_det=50,
        num_map=10,
        use_rescore=True,
        use_causal_history_control=False,
        forbid_gt_history_input=False,
        causal_history_len=1,
        causal_history_source="gt_history",
        reset_history_at_scene_start=True,
        audit_history_source=False,
        nusc_dataroot="data/nuscenes/",
        use_smm=False,
        use_real_mamba=False,
        allow_mamba_fallback=True,
        require_real_mamba=False,
        smm_cfg=None,
        enable_horizon_phase_mamba=False,
        horizon_phase_mamba_cfg=None,
        enable_phase_loss=False,
        enable_far_loss=False,
        enable_longitudinal_loss=False,
        lambda_phase=0.05,
        lambda_far=0.10,
        lambda_longitudinal=0.10,
        phase_v_stop=0.5,
        phase_a_thr=0.5,
        planning_dt=0.5,
        enable_rcpc=False,
        rcpc_cfg=None,
        enable_dense_risk_scorer=False,
        dense_risk_cfg=None,
        enable_selective_pareto_controller=False,
        selective_pareto_cfg=None,
        smm_observability=None,

    ):
        super(MotionPlanningHeadroboAD_6s, self).__init__()
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode

        self.decouple_attn = decouple_attn
        self.operation_order = operation_order
        # Temporal state is per-worker runtime cache. Keep it out of module
        # buffers so DDP does not try to synchronize zero-batch placeholders.
        self.last_planning_classification = torch.zeros(0, 1, 18)
        self.last_planning_prediction = torch.zeros(0, 1, 18, ego_fut_ts, 2)
        self.last_final_planning_prediction = torch.zeros(0, ego_fut_ts, 2)
        self.last_plan_query = torch.zeros(0, 1, 18, embed_dims)
        self.last_ego_cmd = torch.zeros(0, 3)
        self.use_causal_history_control = bool(use_causal_history_control)
        self.forbid_gt_history_input = bool(forbid_gt_history_input)
        self.causal_history_len = int(causal_history_len)
        self.causal_history_source = causal_history_source
        self.reset_history_at_scene_start = bool(reset_history_at_scene_start)
        self.audit_history_source = bool(audit_history_source)
        if self.use_causal_history_control and self.causal_history_len not in (1, 4):
            raise ValueError(
                "Causal MomAD-6s currently supports causal_history_len in {1, 4}"
            )
        if self.use_causal_history_control and self.causal_history_source != "model_prediction":
            raise ValueError(
                "Causal history control requires causal_history_source='model_prediction'"
            )
        self._causal_history_cache = {}
        self._causal_sequence_history_cache = CausalHistoryCache()
        self._causal_cache_write_order = 0
        self._causal_last_local_slot_keys = []
        self._causal_history_audit_records = []
        self._causal_history_audit_max_records = 4096
        self._causal_loss_gt_target_used = False
        self._causal_smm_history_query = None
        self._causal_smm_history_score = None
        self._causal_smm_history_valid_mask = None
        self._causal_smm_reliability_features = None
        self._causal_smm_reliability_availability = None
        self._rcpc_calibration_features = None
        self._rcpc_calibration_availability = None
        self._rcpc_calibration_records = []
        self._causal_smm_reliability_feature_records = []
        self._causal_smm_history_retrieval_audit = []
        self._latest_causal_history_batch_audit_records = []
        self.nusc = NuScenes(version='v1.0-trainval', dataroot=nusc_dataroot, verbose=True)
        self.use_rescore = use_rescore
        self.use_smm = bool(use_smm)
        self.use_real_mamba = bool(use_real_mamba)
        self.allow_mamba_fallback = bool(allow_mamba_fallback)
        self.require_real_mamba = bool(require_real_mamba)
        self.smm_cfg = copy.deepcopy(smm_cfg) if smm_cfg is not None else {}
        self.causal_smm_adapter = None
        self._pending_causal_smm_adapter_cfg = None
        self.enable_horizon_phase_mamba = bool(enable_horizon_phase_mamba)
        self.enable_phase_loss = bool(enable_phase_loss)
        self.enable_far_loss = bool(enable_far_loss)
        self.enable_longitudinal_loss = bool(enable_longitudinal_loss)
        self.lambda_phase = float(lambda_phase)
        self.lambda_far = float(lambda_far)
        self.lambda_longitudinal = float(lambda_longitudinal)
        self.phase_v_stop = float(phase_v_stop)
        self.phase_a_thr = float(phase_a_thr)
        self.planning_dt = float(planning_dt)
        self.enable_rcpc = bool(enable_rcpc)
        self.rcpc_cfg = copy.deepcopy(rcpc_cfg) if rcpc_cfg is not None else {}
        self.enable_dense_risk_scorer = bool(enable_dense_risk_scorer)
        self.dense_risk_cfg = (
            copy.deepcopy(dense_risk_cfg) if dense_risk_cfg is not None else {}
        )
        self.enable_selective_pareto_controller = bool(
            enable_selective_pareto_controller
        )
        self.selective_history_isolation = bool(
            self.enable_selective_pareto_controller
        )
        self.selective_pareto_cfg = (
            copy.deepcopy(selective_pareto_cfg)
            if selective_pareto_cfg is not None
            else {}
        )
        self.smm_observability_cfg = (
            copy.deepcopy(smm_observability)
            if smm_observability is not None
            else {}
        )
        self.smm_observability = SMMObservabilityObserver(
            **self.smm_observability_cfg
        )
        self.horizon_phase_mamba_cfg = (
            copy.deepcopy(horizon_phase_mamba_cfg)
            if horizon_phase_mamba_cfg is not None
            else {}
        )
        if self.use_smm:
            if not self.use_causal_history_control:
                raise ValueError("C1 SMM requires use_causal_history_control=True")
            if not self.forbid_gt_history_input:
                raise ValueError("C1 SMM requires forbid_gt_history_input=True")
            adapter_cfg = dict(
                embed_dims=embed_dims,
                num_modes=18,
                history_len=self.causal_history_len,
                use_real_mamba=self.use_real_mamba,
                require_real_mamba=self.require_real_mamba,
                allow_mamba_fallback=self.allow_mamba_fallback,
            )
            adapter_cfg.update(self.smm_cfg)
            self._pending_causal_smm_adapter_cfg = adapter_cfg
        self.c3a_horizon_adapter = None
        if self.enable_horizon_phase_mamba:
            if not self.use_causal_history_control or self.causal_history_len != 1:
                raise ValueError("C3-A requires strict causal H1 history control")
            if not self.use_smm:
                raise ValueError("C3-A is defined as C1 H1 Causal SMM + Horizon Mamba")
            c3a_cfg = dict(
                embed_dims=embed_dims,
                horizon_steps=ego_fut_ts,
                num_commands=3,
                num_modes=ego_fut_mode,
                dt=self.planning_dt,
                require_real_mamba=True,
                allow_mamba_fallback=False,
            )
            c3a_cfg.update(self.horizon_phase_mamba_cfg)
            self.c3a_horizon_adapter = C3AHorizonPlanningAdapter(**c3a_cfg)
        self.rcpc_adapter = None
        if self.enable_rcpc:
            if self.enable_horizon_phase_mamba:
                raise ValueError("C4 RCPC and C3-A are mutually exclusive core modules")
            if self.enable_dense_risk_scorer:
                raise ValueError("legacy RCPC and the joint dense scorer are exclusive")
            if self.enable_selective_pareto_controller:
                raise ValueError("legacy RCPC and CSPC are exclusive")
            if not self.use_causal_history_control or self.causal_history_len != 1:
                raise ValueError("C4 RCPC requires strict causal H1 history control")
            if not self.forbid_gt_history_input:
                raise ValueError("C4 RCPC requires GT-free planning history")
            if not self.use_smm:
                raise ValueError("C4 RCPC is defined on the C1 H1 Causal SMM backbone")
            rcpc_adapter_cfg = dict(
                embed_dims=embed_dims,
                horizon_steps=ego_fut_ts,
                num_commands=3,
                num_modes=ego_fut_mode,
                dt=self.planning_dt,
            )
            rcpc_adapter_cfg.update(self.rcpc_cfg)
            self.rcpc_adapter = RCPCPlanningAdapter(**rcpc_adapter_cfg)
        self.dense_risk_scorer = None
        if self.enable_dense_risk_scorer:
            if self.enable_selective_pareto_controller:
                raise ValueError("unrestricted CDRS and CSPC are exclusive")
            if not self.use_causal_history_control or self.causal_history_len != 1:
                raise ValueError("CDRS requires strict causal H1 history")
            if not self.forbid_gt_history_input:
                raise ValueError("CDRS requires GT-free planning history")
            if not self.use_smm:
                raise ValueError("CDRS is defined on the C1 causal SMM backbone")
            dense_risk_adapter_cfg = dict(
                embed_dims=embed_dims,
                horizon_steps=ego_fut_ts,
                num_commands=3,
                num_modes=ego_fut_mode,
                dt=self.planning_dt,
            )
            dense_risk_adapter_cfg.update(self.dense_risk_cfg)
            self.dense_risk_scorer = CDRSPlanningAdapter(
                **dense_risk_adapter_cfg
            )
        self.selective_pareto_controller = None
        if self.enable_selective_pareto_controller:
            if not self.use_causal_history_control or self.causal_history_len != 1:
                raise ValueError("CSPC requires strict causal H1 history")
            if not self.forbid_gt_history_input:
                raise ValueError("CSPC requires GT-free planning history")
            if not self.use_smm:
                raise ValueError("CSPC is defined on the C1 causal SMM backbone")
            selective_cfg = dict(
                embed_dims=embed_dims,
                horizon_steps=ego_fut_ts,
                num_commands=3,
                num_modes=ego_fut_mode,
                dt=self.planning_dt,
            )
            selective_cfg.update(self.selective_pareto_cfg)
            self.selective_pareto_controller = CSPCPlanningAdapter(
                **selective_cfg
            )
        # =========== build modules ===========
        def build(cfg, registry):
            if cfg is None:
                return None
            return build_from_cfg(cfg, registry)

        self.instance_queue = build(instance_queue, PLUGIN_LAYERS)
        self.motion_sampler = build(motion_sampler, BBOX_SAMPLERS)
        self.planning_sampler = build(planning_sampler, BBOX_SAMPLERS)
        self.motion_decoder = build(motion_decoder, BBOX_CODERS)
        self.planning_decoder = build(planning_decoder, BBOX_CODERS)
        self.op_config_map = {
            "temp_gnn": [temp_graph_model, ATTENTION],
            "gnn": [graph_model, ATTENTION],
            "cross_gnn": [cross_graph_model, ATTENTION],
            "norm": [norm_layer, NORM_LAYERS],
            "ffn": [ffn, FEEDFORWARD_NETWORK],
            "refine": [refine_layer, PLUGIN_LAYERS],
        }
        self.layers = nn.ModuleList(
            [
                build(*self.op_config_map.get(op, [None, None]))
                for op in self.operation_order
            ]
        )
        self.embed_dims = embed_dims
        self.next_token_predictor = NextTokenPredictor(embed_dims, 128)

        if self.decouple_attn:
            self.fc_before = nn.Linear(
                self.embed_dims, self.embed_dims * 2, bias=False
            )
            self.fc_after = nn.Linear(
                self.embed_dims * 2, self.embed_dims, bias=False
            )
        else:
            self.fc_before = nn.Identity()
            self.fc_after = nn.Identity()

        self.motion_loss_cls = build_loss(motion_loss_cls)
        self.motion_loss_reg = build_loss(motion_loss_reg)
        self.plan_loss_cls = build_loss(plan_loss_cls)
        self.plan_loss_reg = build_loss(plan_loss_reg)
        self.plan_loss_status = build_loss(plan_loss_status)
        self.last_op = ""

        # motion init
        motion_anchor = np.load(motion_anchor)
        self.motion_anchor = nn.Parameter(
            torch.tensor(motion_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.motion_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        # plan anchor init
        plan_anchor = np.load(plan_anchor)
        self.plan_anchor = nn.Parameter(
            torch.tensor(plan_anchor, dtype=torch.float32),
            requires_grad=False,
        )
        self.plan_anchor_encoder = nn.Sequential(
            *linear_relu_ln(embed_dims, 1, 1),
            Linear(embed_dims, embed_dims),
        )

        self.num_det = num_det
        self.num_map = num_map
        #self.sa_atten_layer = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.refine_2th_layer = MotionPlanning2thRefinementModule(embed_dims=256, ego_fut_ts=12, ego_fut_mode=6)
        if self._pending_causal_smm_adapter_cfg is not None:
            self.causal_smm_adapter = CausalSMMAdapter(
                **self._pending_causal_smm_adapter_cfg
            )


    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

        for m in self.modules():
            #import pdb;pdb.set_trace()
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _get_causal_rank_id(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_rank())
        return int(os.environ.get("RANK", 0))

    def _get_causal_world_size(self):
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            return int(torch.distributed.get_world_size())
        return int(os.environ.get("WORLD_SIZE", 1))

    def _ensure_causal_history_token_state(self, batch_size):
        if len(self._causal_last_local_slot_keys) != batch_size:
            self._causal_last_local_slot_keys = [None] * batch_size

    def _causal_entry_detached(self, entry):
        tensor_keys = [
            "last_planning_classification",
            "last_planning_prediction",
            "last_plan_query",
            "last_final_planning_prediction",
            "last_ego_cmd",
        ]
        return all(
            isinstance(entry.get(key), torch.Tensor)
            and not entry[key].requires_grad
            for key in tensor_keys
        )

    def _causal_entry_shapes_valid(self, entry):
        return (
            list(entry["last_planning_classification"].shape) == [1, 18]
            and list(entry["last_planning_prediction"].shape)
            == [1, 18, self.ego_fut_ts, 2]
            and list(entry["last_plan_query"].shape)
            == [1, 18, self.embed_dims]
            and list(entry["last_final_planning_prediction"].shape)
            == [self.ego_fut_ts, 2]
            and list(entry["last_ego_cmd"].shape) == [3]
        )

    def _copy_causal_entry_to_row(self, row_idx, entry, device, dtype):
        self.last_planning_classification[row_idx] = entry[
            "last_planning_classification"
        ].to(device=device, dtype=dtype)
        self.last_planning_prediction[row_idx] = entry[
            "last_planning_prediction"
        ].to(device=device, dtype=dtype)
        self.last_plan_query[row_idx] = entry["last_plan_query"].to(
            device=device, dtype=dtype
        )
        self.last_final_planning_prediction[row_idx] = entry[
            "last_final_planning_prediction"
        ].to(device=device, dtype=dtype)
        self.last_ego_cmd[row_idx] = entry["last_ego_cmd"].to(
            device=device, dtype=dtype
        )

    def _copy_causal_sequence_entry_to_row(self, row_idx, entry, device, dtype):
        self.last_planning_classification[row_idx] = entry.candidate_logits.to(
            device=device, dtype=dtype
        )
        self.last_planning_prediction[row_idx] = entry.candidate_trajectory.to(
            device=device, dtype=dtype
        )
        self.last_plan_query[row_idx] = entry.plan_query.to(
            device=device, dtype=dtype
        )
        self.last_final_planning_prediction[row_idx] = entry.selected_planning.to(
            device=device, dtype=dtype
        )
        self.last_ego_cmd[row_idx] = entry.metadata["last_ego_cmd"].to(
            device=device, dtype=dtype
        )

    def _sample_record_timestamp_sec(self, sample_record):
        timestamp = float(sample_record.get("timestamp", 0.0))
        return timestamp / 1e6 if timestamp > 1e12 else timestamp

    def _sample_record_pose(self, sample_record):
        sample_data = self.nusc.get("sample_data", sample_record["data"]["LIDAR_TOP"])
        ego_pose = self.nusc.get("ego_pose", sample_data["ego_pose_token"])
        translation = ego_pose["translation"]
        rotation = ego_pose["rotation"]
        return translation, quaternion_yaw(rotation), rotation

    def _ego_status_for_cache(self, metas, idx, device=None, dtype=None):
        ego_status = metas.get("ego_status", None)
        if ego_status is None:
            tensor = torch.zeros(10)
        else:
            tensor = ego_status[idx].detach().clone()
        if device is not None:
            tensor = tensor.to(device=device)
        if dtype is not None:
            tensor = tensor.to(dtype=dtype)
        return tensor.detach()

    def _make_causal_sequence_entry(
        self,
        idx,
        token,
        prev_token,
        scene_token,
        sample_record,
        metas,
        rank_id,
    ):
        translation, yaw, rotation = self._sample_record_pose(sample_record)
        ego_status = self._ego_status_for_cache(metas, idx)
        return CausalHistoryEntry(
            sample_token=token,
            prev_token=prev_token,
            scene_token=scene_token,
            rank_id=int(rank_id),
            timestamp=self._sample_record_timestamp_sec(sample_record),
            cache_write_order=int(self._causal_cache_write_order),
            history_source="model_prediction",
            plan_query=self.last_plan_query[idx].detach().clone(),
            candidate_trajectory=self.last_planning_prediction[idx].detach().clone(),
            selected_planning=self.last_final_planning_prediction[idx].detach().clone(),
            candidate_logits=self.last_planning_classification[idx].detach().clone(),
            ego_translation=(
                float(translation[0]),
                float(translation[1]),
                float(translation[2]),
            ),
            ego_yaw=float(yaw),
            ego_velocity=(
                float(ego_status[7].item()) if ego_status.numel() > 7 else 0.0,
                float(ego_status[6].item()) if ego_status.numel() > 6 else 0.0,
            ),
            ego_status=ego_status,
            metadata={
                "last_ego_cmd": self.last_ego_cmd[idx].detach().clone(),
                "ego2global_rotation": [float(v) for v in rotation],
                "trajectory_dt_sec": 0.5,
                "trajectory_horizon_steps": int(self.ego_fut_ts),
            },
        )

    def _retrospective_reliability_feature_for_entry(
        self,
        entry,
        age,
        current_sample_record,
        metas,
        batch_idx,
    ):
        """Build strict-causal CR-v1 calibration features for one history entry.

        Values use only historical model prediction plus the current observed
        ego pose/status. GT future trajectory is not read here.
        """
        values = np.zeros(6, dtype=np.float32)
        availability = np.zeros(4, dtype=np.float32)
        record = {
            "age": int(age),
            "history_token": entry.sample_token,
            "feature_available": False,
            "uses_gt_future": False,
            "normalization_source": "fixed_physical_scales_in_reliability_gate",
        }
        try:
            current_translation, current_yaw, _ = self._sample_record_pose(
                current_sample_record
            )
            current_pose = EgoPose2D(
                translation_xy=np.asarray(current_translation[:2], dtype=np.float64),
                yaw=float(current_yaw),
                timestamp=self._sample_record_timestamp_sec(current_sample_record),
            )
            hist_pose = EgoPose2D(
                translation_xy=np.asarray(entry.ego_translation[:2], dtype=np.float64),
                yaw=float(entry.ego_yaw),
                timestamp=float(entry.timestamp),
            )
            current_ego_status = self._ego_status_for_cache(metas, batch_idx).cpu().numpy()
            selected_planning = entry.selected_planning.detach().float().cpu().numpy()
            features = retrospective_calibration_features(
                historical_traj_cumulative=selected_planning,
                age=int(age),
                hist_pose=hist_pose,
                current_pose=current_pose,
                current_ego_status=current_ego_status,
            )
            obs_velocity = observed_velocity_xy_from_ego_status(current_ego_status)
            current_speed = (
                float(np.linalg.norm(obs_velocity))
                if obs_velocity is not None
                else 0.0
            )
            values[4] = current_speed
            values[5] = float(age) * 0.5
            if not features.get("available", False):
                record.update(features)
                return values, availability, record

            e_pos = float(features.get("e_pos", 0.0))
            e_vel_raw = features.get("e_vel", None)
            e_yaw = float(features.get("e_yaw", 0.0))
            e_progress = float(features.get("e_progress", 0.0))
            values[0] = e_pos
            values[1] = float(e_vel_raw) if e_vel_raw is not None else 0.0
            values[2] = e_yaw
            values[3] = e_progress
            availability[0] = 1.0
            availability[1] = 1.0 if e_vel_raw is not None else 0.0
            pred_velocity = np.asarray(
                features.get("pred_current_velocity_xy", [0.0, 0.0]),
                dtype=np.float64,
            )
            pred_speed = float(np.linalg.norm(pred_velocity))
            # Heading is unstable at near-zero speed, so expose the mask.
            availability[2] = 1.0 if current_speed > 0.5 and pred_speed > 0.5 else 0.0
            availability[3] = 1.0
            record.update(features)
            record.update(
                {
                    "feature_available": True,
                    "current_speed": current_speed,
                    "pred_speed": pred_speed,
                    "feature_values": [float(v) for v in values.tolist()],
                    "availability_mask": [float(v) for v in availability.tolist()],
                }
            )
            return values, availability, record
        except Exception as exc:
            record["feature_error"] = repr(exc)
            return values, availability, record

    def _retrospective_reliability_feature_for_h1_entry(
        self,
        entry,
        current_sample_record,
        metas,
        batch_idx,
    ):
        """Build the RCPC calibration token from the causal t-1 cache."""
        values = np.zeros(6, dtype=np.float32)
        availability = np.zeros(4, dtype=np.float32)
        record = {
            "age": 1,
            "feature_available": False,
            "uses_gt_future": False,
            "history_token": entry.get("sample_token"),
        }
        required = (
            "ego_translation",
            "ego_yaw",
            "last_final_planning_prediction",
        )
        if any(key not in entry for key in required):
            record["reason"] = "h1_alignment_metadata_missing"
            return values, availability, record
        try:
            current_translation, current_yaw, _ = self._sample_record_pose(
                current_sample_record
            )
            current_pose = EgoPose2D(
                translation_xy=np.asarray(current_translation[:2], dtype=np.float64),
                yaw=float(current_yaw),
                timestamp=self._sample_record_timestamp_sec(current_sample_record),
            )
            history_pose = EgoPose2D(
                translation_xy=np.asarray(entry["ego_translation"][:2], dtype=np.float64),
                yaw=float(entry["ego_yaw"]),
                timestamp=float(entry.get("timestamp", 0.0)),
            )
            current_ego_status = self._ego_status_for_cache(
                metas, batch_idx
            ).cpu().numpy()
            features = retrospective_calibration_features(
                historical_traj_cumulative=entry[
                    "last_final_planning_prediction"
                ].detach().float().cpu().numpy(),
                age=1,
                hist_pose=history_pose,
                current_pose=current_pose,
                current_ego_status=current_ego_status,
            )
            observed_velocity = observed_velocity_xy_from_ego_status(
                current_ego_status
            )
            current_speed = (
                float(np.linalg.norm(observed_velocity))
                if observed_velocity is not None
                else 0.0
            )
            values[4] = current_speed
            values[5] = 0.5
            if not features.get("available", False):
                record.update(features)
                return values, availability, record
            predicted_velocity = np.asarray(
                features.get("pred_current_velocity_xy", [0.0, 0.0]),
                dtype=np.float64,
            )
            predicted_speed = float(np.linalg.norm(predicted_velocity))
            values[0] = float(features.get("e_pos", 0.0))
            e_vel = features.get("e_vel", None)
            values[1] = float(e_vel) if e_vel is not None else 0.0
            values[2] = float(features.get("e_yaw", 0.0))
            values[3] = float(features.get("e_progress", 0.0))
            availability[0] = 1.0
            availability[1] = 1.0 if e_vel is not None else 0.0
            availability[2] = (
                1.0 if current_speed > 0.5 and predicted_speed > 0.5 else 0.0
            )
            availability[3] = 1.0
            record.update(features)
            record.update(
                {
                    "feature_available": True,
                    "current_speed": current_speed,
                    "predicted_speed": predicted_speed,
                    "feature_values": [float(value) for value in values.tolist()],
                    "availability_mask": [
                        float(value) for value in availability.tolist()
                    ],
                }
            )
            return values, availability, record
        except Exception as exc:
            record["feature_error"] = repr(exc)
            return values, availability, record

    def _prev_token_getter(self, token):
        if not token:
            return ""
        return self.nusc.get("sample", token)["prev"]

    def _prune_causal_sequence_cache(self):
        max_entries = max(4096, int(self.causal_history_len) * 1024)
        entries = getattr(self._causal_sequence_history_cache, "_entries", None)
        if entries is None or len(entries) <= max_entries:
            return
        ordered = sorted(
            entries.items(),
            key=lambda item: item[1].cache_write_order,
        )
        for key, _ in ordered[: len(entries) - max_entries]:
            entries.pop(key, None)

    def _prepare_causal_sequence_history_for_batch(
        self,
        batch_size,
        device,
        dtype,
        metas,
        sample_tokens,
        sample_records,
        prev_tokens,
        scene_tokens,
    ):
        current_rank = self._get_causal_rank_id()
        self._zero_planning_history_rows(
            torch.ones(batch_size, device=device, dtype=torch.bool)
        )
        h = int(self.causal_history_len)
        history_query = torch.zeros(
            batch_size, h, 18, self.embed_dims, device=device, dtype=dtype
        )
        history_score = torch.zeros(batch_size, h, 18, device=device, dtype=dtype)
        history_valid_mask = torch.zeros(batch_size, h, device=device, dtype=torch.bool)
        reliability_features = torch.zeros(
            batch_size, h, 6, device=device, dtype=dtype
        )
        reliability_availability = torch.zeros(
            batch_size, h, 4, device=device, dtype=dtype
        )
        history_valid_values = []
        audit_records = []
        reliability_feature_records = []
        retriever = CausalHistorySequenceRetriever(
            self._causal_sequence_history_cache, self._prev_token_getter
        )
        for idx, token in enumerate(sample_tokens):
            scene_token = scene_tokens[idx]
            prev_token = prev_tokens[idx]
            current_timestamp = self._sample_record_timestamp_sec(sample_records[idx])
            result = retriever.retrieve(
                current_sample_token=token,
                current_prev_token=prev_token,
                current_scene_token=scene_token,
                current_rank_id=current_rank,
                requested_history_len=h,
                current_timestamp=current_timestamp,
                current_causal_order=self._causal_cache_write_order,
            )
            l_eff = int(result.effective_history_len)
            source_timestamp = (
                float(result.entries_oldest_to_newest[-1].timestamp)
                if l_eff > 0
                else None
            )
            if l_eff > 0:
                start = h - l_eff
                for local_pos, entry in enumerate(result.entries_oldest_to_newest):
                    target_pos = start + local_pos
                    history_query[idx, target_pos] = entry.plan_query.to(
                        device=device, dtype=dtype
                    )[0]
                    history_score[idx, target_pos] = entry.candidate_logits.to(
                        device=device, dtype=dtype
                    )[0]
                    history_valid_mask[idx, target_pos] = True
                    actual_age = int(l_eff - local_pos)
                    feat_values, feat_avail, feat_record = (
                        self._retrospective_reliability_feature_for_entry(
                            entry,
                            actual_age,
                            sample_records[idx],
                            metas,
                            idx,
                        )
                    )
                    reliability_features[idx, target_pos] = torch.as_tensor(
                        feat_values, device=device, dtype=dtype
                    )
                    reliability_availability[idx, target_pos] = torch.as_tensor(
                        feat_avail, device=device, dtype=dtype
                    )
                    feat_record.update(
                        {
                            "sample_token": token,
                            "scene_token": scene_token,
                            "target_history_slot": int(target_pos),
                            "actual_age": actual_age,
                        }
                    )
                    reliability_feature_records.append(feat_record)
                self._copy_causal_sequence_entry_to_row(
                    idx,
                    result.entries_oldest_to_newest[-1],
                    device=device,
                    dtype=dtype,
                )
            history_valid_values.append(l_eff > 0)
            miss_reason = result.stop_reason if l_eff < h else None
            audit_records.append(
                {
                    "sample_token": token,
                    "prev_token": prev_token,
                    "scene_token": scene_token,
                    "timestamp": float(current_timestamp),
                    "is_scene_start": bool(prev_token == ""),
                    "history_cache_hit": bool(l_eff > 0),
                    "history_cache_miss_reason": miss_reason,
                    "history_source": "model_prediction" if l_eff > 0 else "zero_reset",
                    "history_cache_key": f"{scene_token}:{prev_token}",
                    "history_source_token": (
                        result.entries_oldest_to_newest[-1].sample_token
                        if l_eff > 0
                        else None
                    ),
                    "history_source_scene_token": scene_token if l_eff > 0 else None,
                    "history_source_timestamp": source_timestamp,
                    "timestamp_delta": (
                        float(current_timestamp - source_timestamp)
                        if source_timestamp is not None
                        else None
                    ),
                    "history_source_rank_id": int(current_rank) if l_eff > 0 else None,
                    "current_rank_id": int(current_rank),
                    "history_prediction_shape": list(
                        self.last_final_planning_prediction[idx].shape
                    ),
                    "current_prediction_shape": [self.ego_fut_ts, 2],
                    "history_detached": all(
                        entry.neural_tensors_detached()
                        for entry in result.entries_oldest_to_newest
                    ),
                    "history_cache_entry_shape_valid": all(
                        entry.shapes_valid() for entry in result.entries_oldest_to_newest
                    ),
                    "gt_history_write_detected": False,
                    "gt_history_read_detected_for_temporal_input": False,
                    "cache_cross_scene_detected": bool(
                        miss_reason == "scene_mismatch"
                    ),
                    "cross_rank_history_entry_detected": bool(
                        miss_reason == "rank_mismatch"
                    ),
                    "prev_token_continuity_verified": bool(l_eff > 0),
                    "loss_gt_target_used": False,
                    "SMM_enabled": bool(self.use_smm),
                    "requested_history_len": h,
                    "effective_history_len": l_eff,
                    "per_age_valid_newest_first": list(
                        result.per_age_valid_newest_first
                    ),
                    "history_tokens_oldest_to_newest": [
                        entry.sample_token for entry in result.entries_oldest_to_newest
                    ],
                    "sequence_stop_reason": result.stop_reason,
                    "sequence_retrieval_audit": result.audit,
                    "alignment_metadata_available": all(
                        entry.ego_translation is not None and entry.ego_yaw is not None
                        for entry in result.entries_oldest_to_newest
                    ),
                    "reliability_feature_records": [
                        rec for rec in reliability_feature_records
                        if rec.get("sample_token") == token
                    ],
                    "beta_values": [1.0] * l_eff,
                }
            )
        self._causal_smm_history_query = history_query
        self._causal_smm_history_score = history_score
        self._causal_smm_history_valid_mask = history_valid_mask
        self._causal_smm_reliability_features = reliability_features
        self._causal_smm_reliability_availability = reliability_availability
        if self.enable_rcpc:
            self._rcpc_calibration_features = reliability_features[:, -1]
            self._rcpc_calibration_availability = reliability_availability[:, -1]
            self._rcpc_calibration_records = list(reliability_feature_records)
        self._causal_smm_reliability_feature_records = reliability_feature_records
        self._causal_smm_history_retrieval_audit = audit_records
        self._latest_causal_history_batch_audit_records = list(audit_records)
        self._append_causal_history_audit(audit_records)
        return torch.tensor(history_valid_values, device=device, dtype=torch.bool)

    def _causal_miss_reason(self, idx, scene_token, prev_token):
        if prev_token == "":
            return "scene_start"
        for cached_scene_token, cached_sample_token in self._causal_history_cache:
            if cached_sample_token == prev_token and cached_scene_token != scene_token:
                return "scene_mismatch"
        if idx < len(self._causal_last_local_slot_keys):
            slot_key = self._causal_last_local_slot_keys[idx]
            if slot_key is not None:
                slot_scene_token, slot_sample_token = slot_key
                if slot_scene_token == scene_token and slot_sample_token != prev_token:
                    return "token_discontinuity"
        return "prev_token_not_in_local_rank_cache"

    def _zero_planning_history_rows(self, row_mask):
        if row_mask.any():
            self.last_planning_classification[row_mask] = torch.zeros_like(
                self.last_planning_classification[row_mask]
            )
            self.last_planning_prediction[row_mask] = torch.zeros_like(
                self.last_planning_prediction[row_mask]
            )
            self.last_plan_query[row_mask] = torch.zeros_like(
                self.last_plan_query[row_mask]
            )
            self.last_final_planning_prediction[row_mask] = torch.zeros_like(
                self.last_final_planning_prediction[row_mask]
            )
            self.last_ego_cmd[row_mask] = torch.zeros_like(self.last_ego_cmd[row_mask])

    def _append_causal_history_audit(self, records):
        if not self.audit_history_source:
            return
        self._causal_history_audit_records.extend(records)
        if len(self._causal_history_audit_records) > self._causal_history_audit_max_records:
            self._causal_history_audit_records = self._causal_history_audit_records[
                -self._causal_history_audit_max_records:
            ]

    def consume_causal_history_audit(self):
        records = self._causal_history_audit_records
        self._causal_history_audit_records = []
        loss_gt_target_used = bool(self._causal_loss_gt_target_used)
        self._causal_loss_gt_target_used = False
        return {
            "records": records,
            "loss_gt_target_used": loss_gt_target_used,
        }

    def consume_c1_smm_audit(self):
        if self.causal_smm_adapter is None:
            return {
                "records": [],
                "SMM_enabled": False,
                "RealMamba_imported": False,
                "RealMamba_instantiated": False,
                "RealMamba_forward_call_count": 0,
                "SMM_forward_call_count": 0,
                "fallback_activated": False,
            }
        audit = self.causal_smm_adapter.consume_audit()
        audit["SMM_enabled"] = True
        return audit

    def _prepare_planning_history_for_batch(self, batch_size, device, dtype, metas):
        if self.last_plan_query.shape[0] != batch_size:
            self.last_planning_classification = torch.zeros(
                batch_size, 1, 18, device=device, dtype=dtype
            )
            self.last_planning_prediction = torch.zeros(
                batch_size, 1, 18, self.ego_fut_ts, 2,
                device=device, dtype=dtype
            )
            self.last_plan_query = torch.zeros(
                batch_size, 1, 18, self.embed_dims,
                device=device, dtype=dtype
            )
            self.last_final_planning_prediction = torch.zeros(
                batch_size, self.ego_fut_ts, 2, device=device, dtype=dtype
            )
            self.last_ego_cmd = torch.zeros(
                batch_size, 3, device=device, dtype=dtype
            )
            if self.use_causal_history_control:
                self._ensure_causal_history_token_state(batch_size)
        else:
            self.last_planning_classification = (
                self.last_planning_classification.to(device=device, dtype=dtype).detach()
            )
            self.last_planning_prediction = (
                self.last_planning_prediction.to(device=device, dtype=dtype).detach()
            )
            self.last_plan_query = self.last_plan_query.to(
                device=device, dtype=dtype
            ).detach()
            self.last_final_planning_prediction = (
                self.last_final_planning_prediction.to(
                    device=device, dtype=dtype
                ).detach()
            )
            self.last_ego_cmd = self.last_ego_cmd.to(
                device=device, dtype=dtype
            ).detach()

        sample_tokens = [item["token"] for item in metas["img_metas"]]
        sample_records = [self.nusc.get("sample", token) for token in sample_tokens]
        prev_tokens = [record["prev"] for record in sample_records]
        scene_tokens = [record["scene_token"] for record in sample_records]
        self._latest_causal_history_batch_audit_records = []
        scene_start = torch.tensor(
            [prev_token == "" for prev_token in prev_tokens],
            device=device,
            dtype=torch.bool,
        )
        if self.enable_rcpc:
            self._rcpc_calibration_features = torch.zeros(
                batch_size, 6, device=device, dtype=dtype
            )
            self._rcpc_calibration_availability = torch.zeros(
                batch_size, 4, device=device, dtype=dtype
            )
            self._rcpc_calibration_records = []
        causal_history_valid = torch.zeros(batch_size, device=device, dtype=torch.bool)
        if self.use_causal_history_control:
            if self.causal_history_len > 1:
                causal_history_valid = self._prepare_causal_sequence_history_for_batch(
                    batch_size,
                    device,
                    dtype,
                    metas,
                    sample_tokens,
                    sample_records,
                    prev_tokens,
                    scene_tokens,
                )
                return sample_tokens, prev_tokens, scene_tokens, causal_history_valid
            self._ensure_causal_history_token_state(batch_size)
            current_rank = self._get_causal_rank_id()
            self._zero_planning_history_rows(
                torch.ones(batch_size, device=device, dtype=torch.bool)
            )
            history_valid_values = []
            audit_records = []
            for idx, token in enumerate(sample_tokens):
                prev_token = prev_tokens[idx]
                scene_token = scene_tokens[idx]
                lookup_key = (scene_token, prev_token)
                cached_entry = (
                    self._causal_history_cache.get(lookup_key)
                    if prev_token != ""
                    else None
                )
                cache_hit = cached_entry is not None
                cached_rank = cached_entry.get("rank_id") if cache_hit else None
                cached_token = cached_entry.get("sample_token") if cache_hit else None
                cached_scene_token = (
                    cached_entry.get("scene_token") if cache_hit else None
                )
                cached_timestamp = cached_entry.get("timestamp") if cache_hit else None
                current_timestamp = self._sample_record_timestamp_sec(sample_records[idx])
                entry_detached = (
                    self._causal_entry_detached(cached_entry) if cache_hit else True
                )
                entry_shapes_valid = (
                    self._causal_entry_shapes_valid(cached_entry)
                    if cache_hit and entry_detached
                    else False
                )
                cross_scene_detected = bool(
                    cache_hit and cached_scene_token != scene_token
                )
                cross_rank_detected = bool(
                    cache_hit and cached_rank is not None and cached_rank != current_rank
                )
                continuity_verified = (
                    cache_hit
                    and prev_token != ""
                    and cached_token == prev_token
                    and cached_scene_token == scene_token
                    and cached_rank == current_rank
                    and entry_detached
                    and entry_shapes_valid
                )
                rcpc_feature_record = None
                if continuity_verified:
                    self._copy_causal_entry_to_row(
                        idx, cached_entry, device=device, dtype=dtype
                    )
                    if self.enable_rcpc:
                        feature_values, feature_availability, rcpc_feature_record = (
                            self._retrospective_reliability_feature_for_h1_entry(
                                cached_entry,
                                sample_records[idx],
                                metas,
                                idx,
                            )
                        )
                        self._rcpc_calibration_features[idx] = torch.as_tensor(
                            feature_values, device=device, dtype=dtype
                        )
                        self._rcpc_calibration_availability[idx] = torch.as_tensor(
                            feature_availability, device=device, dtype=dtype
                        )
                        rcpc_feature_record.update(
                            {
                                "sample_token": token,
                                "scene_token": scene_token,
                            }
                        )
                        self._rcpc_calibration_records.append(rcpc_feature_record)
                history_valid_values.append(continuity_verified)
                source = "model_prediction" if continuity_verified else "zero_reset"
                miss_reason = (
                    None
                    if continuity_verified
                    else self._causal_miss_reason(idx, scene_token, prev_token)
                )
                audit_records.append(
                    {
                        "sample_token": token,
                        "prev_token": prev_token,
                        "scene_token": scene_token,
                        "timestamp": float(current_timestamp),
                        "is_scene_start": bool(prev_token == ""),
                        "history_cache_hit": bool(continuity_verified),
                        "history_cache_miss_reason": miss_reason,
                        "history_source": source,
                        "history_cache_key": f"{scene_token}:{prev_token}",
                        "history_source_token": cached_token if continuity_verified else None,
                        "history_source_scene_token": (
                            cached_scene_token if continuity_verified else None
                        ),
                        "history_source_timestamp": (
                            float(cached_timestamp)
                            if continuity_verified and cached_timestamp is not None
                            else None
                        ),
                        "timestamp_delta": (
                            float(current_timestamp - cached_timestamp)
                            if continuity_verified and cached_timestamp is not None
                            else None
                        ),
                        "history_source_rank_id": (
                            int(cached_rank) if continuity_verified else None
                        ),
                        "current_rank_id": int(current_rank),
                        "history_prediction_shape": list(
                            self.last_final_planning_prediction[idx].shape
                        ),
                        "current_prediction_shape": [self.ego_fut_ts, 2],
                        "history_detached": bool(entry_detached),
                        "history_cache_entry_shape_valid": bool(
                            entry_shapes_valid or not cache_hit
                        ),
                        "gt_history_write_detected": False,
                        "gt_history_read_detected_for_temporal_input": False,
                        "cache_cross_scene_detected": cross_scene_detected,
                        "cross_rank_history_entry_detected": cross_rank_detected,
                        "prev_token_continuity_verified": bool(continuity_verified),
                        "loss_gt_target_used": False,
                        "SMM_enabled": bool(self.use_smm),
                        "rcpc_calibration": rcpc_feature_record,
                    }
                )
            causal_history_valid = torch.tensor(
                history_valid_values, device=device, dtype=torch.bool
            )
            self._latest_causal_history_batch_audit_records = list(audit_records)
            self._append_causal_history_audit(audit_records)
        elif scene_start.any():
            self._zero_planning_history_rows(scene_start)
        return sample_tokens, prev_tokens, scene_tokens, causal_history_valid

    def get_motion_anchor(
        self,
        classification,
        prediction,
    ):
        cls_ids = classification.argmax(dim=-1)
        motion_anchor = self.motion_anchor[cls_ids]
        prediction = prediction.detach()
        return self._agent2lidar(motion_anchor, prediction)

    def _agent2lidar(self, trajs, boxes):
        yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
        cos_yaw = torch.cos(yaw)
        sin_yaw = torch.sin(yaw)
        rot_mat_T = torch.stack(
            [
                torch.stack([cos_yaw, sin_yaw]),
                torch.stack([-sin_yaw, cos_yaw]),
            ]
        )

        trajs_lidar = torch.einsum('abcij,jkab->abcik', trajs, rot_mat_T)
        return trajs_lidar


    def rescore(
        self,
        plan_cls,
        plan_reg,
        motion_cls,
        motion_reg,
        det_anchors,
        det_confidence,
        score_thresh=0.5,
        static_dis_thresh=0.5,
        dim_scale=1.1,
        num_motion_mode=1,
        offset=0.5,
    ):

        def cat_with_zero(traj):
            zeros = traj.new_zeros(traj.shape[:-2] + (1, 2))
            traj_cat = torch.cat([zeros, traj], dim=-2)
            return traj_cat

        def get_yaw(traj, start_yaw=np.pi/2):
            yaw = traj.new_zeros(traj.shape[:-1])
            yaw[..., 1:-1] = torch.atan2(
                traj[..., 2:, 1] - traj[..., :-2, 1],
                traj[..., 2:, 0] - traj[..., :-2, 0],
            )
            yaw[..., -1] = torch.atan2(
                traj[..., -1, 1] - traj[..., -2, 1],
                traj[..., -1, 0] - traj[..., -2, 0],
            )
            yaw[..., 0] = start_yaw
            # for static object, estimated future yaw would be unstable
            start = traj[..., 0, :]
            end = traj[..., -1, :]
            dist = torch.linalg.norm(end - start, dim=-1)
            mask = dist < static_dis_thresh
            start_yaw = yaw[..., 0].unsqueeze(-1)
            yaw = torch.where(
                mask.unsqueeze(-1),
                start_yaw,
                yaw,
            )
            return yaw.unsqueeze(-1)

        ## ego
        bs = plan_reg.shape[0]
        plan_reg_cat = cat_with_zero(plan_reg)
        ego_box = det_anchors.new_zeros(bs, self.ego_fut_mode, self.ego_fut_ts + 1, 7)
        ego_box[..., [X, Y]] = plan_reg_cat
        ego_box[..., [W, L, H]] = ego_box.new_tensor([4.08, 1.73, 1.56]) * dim_scale
        ego_box[..., [YAW]] = get_yaw(plan_reg_cat)

        ## motion
        motion_reg = motion_reg[..., :self.ego_fut_ts, :].cumsum(-2)
        motion_reg = cat_with_zero(motion_reg) + det_anchors[:, :, None, None, :2]
        _, motion_mode_idx = torch.topk(motion_cls, num_motion_mode, dim=-1)
        motion_mode_idx = motion_mode_idx[..., None, None].repeat(1, 1, 1, self.ego_fut_ts + 1, 2)
        motion_reg = torch.gather(motion_reg, 2, motion_mode_idx)

        motion_box = motion_reg.new_zeros(motion_reg.shape[:-1] + (7,))
        motion_box[..., [X, Y]] = motion_reg
        motion_box[..., [W, L, H]] = det_anchors[..., None, None, [W, L, H]].exp()
        box_yaw = torch.atan2(
            det_anchors[..., SIN_YAW],
            det_anchors[..., COS_YAW],
        )
        motion_box[..., [YAW]] = get_yaw(motion_reg, box_yaw.unsqueeze(-1))

        filter_mask = det_confidence < score_thresh
        motion_box[filter_mask] = 1e6

        ego_box = ego_box[..., 1:, :]
        motion_box = motion_box[..., 1:, :]

        bs, num_ego_mode, ts, _ = ego_box.shape
        bs, num_anchor, num_motion_mode, ts, _ = motion_box.shape
        ego_box = ego_box[:, None, None].repeat(1, num_anchor, num_motion_mode, 1, 1, 1).flatten(0, -2)
        motion_box = motion_box.unsqueeze(3).repeat(1, 1, 1, num_ego_mode, 1, 1).flatten(0, -2)

        ego_box[..., X] += offset * torch.cos(ego_box[..., YAW])
        ego_box[..., Y] += offset * torch.sin(ego_box[..., YAW])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls


    def graph_model(
        self,
        index,
        query,
        key=None,
        value=None,
        query_pos=None,
        key_pos=None,
        **kwargs,
    ):
        if self.decouple_attn:
            query = torch.cat([query, query_pos], dim=-1)
            if key is not None:
                key = torch.cat([key, key_pos], dim=-1)
            query_pos, key_pos = None, None
        if value is not None:
            value = self.fc_before(value)
        return self.fc_after(
            self.layers[index](
                query,
                key,
                value,
                query_pos=query_pos,
                key_pos=key_pos,
                **kwargs,
            )
        )
    def sa_atten(self, plan_query, last_query):
        #import pdb;pdb.set_trace()
        plan_query = self.sa_atten_layer(torch.cat([plan_query,last_query],dim=-1).permute(0,3,1,2))
        return plan_query.permute(0,2,3,1)
    def select_last_final_planning_prediction(self,planning_classification,planning_prediction,metas):
        #print("hhhhhhhhhhhhhhhhhhhhhh",len(planning_classification))
        #import pdb; pdb.set_trace()
        bs = planning_classification[0].shape[0]
        #bs = planning_classification_cmd.shape[0] #3
        planning_classification_cmd = planning_classification[-1].reshape(bs, 3, self.ego_fut_mode)#[3, 3, 6]
        planning_prediction_cmd = planning_prediction[-1].reshape(bs, 3, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)#[3, 3, 6, 12, 2]
        bs_indices = torch.arange(bs, device=planning_classification_cmd.device) #tensor([0, 1, 2], device='cuda:0')
        gt_ego_fut_cmd = metas["gt_ego_fut_cmd"].argmax(dim=-1)#tensor([2, 2, 2], device='cuda:0')
        planning_classification_cmd_select = planning_classification_cmd[bs_indices, gt_ego_fut_cmd]#torch.Size([3, 6])
        planning_prediction_cmd_select = planning_prediction_cmd[bs_indices, gt_ego_fut_cmd]#torch.Size([3, 6, 12, 2])

        mode_idx = planning_classification_cmd_select.argmax(dim=-1)#tensor([5, 5, 0], device='cuda:0')
        last_final_planning_prediction = planning_prediction_cmd_select[bs_indices, mode_idx] #torch.Size([3, 12, 2])
        return last_final_planning_prediction
    def select_currect_best_planning(self, last_final_planning_prediction, planning_prediction, current_ego_cmd, last_ego_cmd):
        # import pdb; pdb.set_trace()
        ego_mask=(current_ego_cmd.argmax(-1) == last_ego_cmd.argmax(-1))
        # Legacy non-causal training writes GT increments into the history
        # buffer, whereas causal training and every inference path write the
        # model-selected cumulative trajectory. Normalize both paths to
        # cumulative coordinates exactly once before matching current modes.
        previous_planning_cumulative = last_final_planning_prediction
        if self.training and not self.use_causal_history_control:
            previous_planning_cumulative = previous_planning_cumulative.cumsum(
                dim=-2
            )
        current_planning_cumulative = planning_prediction[0].squeeze(1).cumsum(
            dim=-2
        )
        distances_euclidean = self.euclidean_distance(
            previous_planning_cumulative.unsqueeze(1).repeat(1, 18, 1, 1),
            current_planning_cumulative,
        )
        min_idx = torch.argmin(distances_euclidean, dim=-1).cpu().numpy().tolist()
        bs = planning_prediction[0].squeeze(1).shape[0]
        #import pdb;pdb.set_trace()
        select_currect_best_planning_prediction = torch.where(ego_mask.view(-1, 1, 1), planning_prediction[0].squeeze(1)[torch.arange(bs), min_idx], torch.full((12, 2), 999.9).repeat(bs, 1, 1).to(ego_mask.device))
        #import pdb;pdb.set_trace()
        return select_currect_best_planning_prediction, min_idx, ego_mask

    def select_former_best_query(self, plan_query, last_ego_cmd, planning_classification):
        last_ego_cmd = last_ego_cmd.repeat_interleave(6, dim=1)
        planning_classification = planning_classification.squeeze(1).sigmoid() * last_ego_cmd

        return planning_classification.argmax(-1)

    def euclidean_distance(self, TA, TB):
        if TA.shape != TB.shape:
            raise ValueError(
                "Trajectory distance expects shape-matched cumulative paths, "
                f"got {tuple(TA.shape)} and {tuple(TB.shape)}"
            )
        distances = torch.sqrt(torch.sum((TA - TB)**2, dim=-1))
        distance_sums = distances.sum(dim=-1)
        return distance_sums
    def dtw_distance(self, TA, TB):
        TA = TA.cumsum(dim=-2)
        TB = TB.cumsum(dim=-2)
        TA_np = TA.detach().cpu().numpy()  # 将tensor转换为numpy数组
        TB_np = TB.detach().cpu().numpy()
        distance, _ = fastdtw(TA_np, TB_np, dist=euclidean)
        return distance
    def forward(
        self,
        det_output,
        map_output,
        feature_maps,
        metas,
        anchor_encoder,
        mask,
        anchor_handler,
    ):

        # =========== det/map feature/anchor ===========
        instance_feature = det_output["instance_feature"]
        #det_output包括 ['clas sification', 'prediction', 'quality', 'instance_feature', 'anchor_embed', 'instance_id']
        #det_output['classification'][0].shape  torch.Size([6, 900, 10]) (cam,anchor,class)
        #det_output['prediction'][0].shape  torch.Size([6, 900, 11]) (cam,anchor,11) 11:{x, y, z, ln w, ln h, ln l, sin yaw, cos yaw, vx, vy, vz}
        #det_output['quality'][0].shape  torch.Size([6, 900, 2]) (cam,anchor,2) 2:centerness,yawness
        #det_output['instance_feature'][0].shape  torch.Size([6, 900, 256]) (cam,anchor,dim)
        #det_output['anchor_embed'][0].shape  torch.Size([6, 900, 256]) (cam,anchor,dim)
        anchor_embed = det_output["anchor_embed"]
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        _, (instance_feature_selected, anchor_embed_selected) = topk(
            det_confidence, self.num_det, instance_feature, anchor_embed
        )
        #instance_feature_selected.shape  torch.Size([6, 50, 256])
        map_instance_feature = map_output["instance_feature"]
        #map_output包括dict_keys(['classification', 'prediction', 'quality', 'instance_feature', 'anchor_embed'])
        #map_output[classification] torch.Size([6, 100, 3])
        #map_output[prediction] torch.Size([6, 100, 40])
        #map_output[quality] [None, None, None, None, None, None]
        #map_output[instance_feature] torch.Size([6, 100, 256])
        #map_output[anchor_embed] torch.Size([6, 100, 256])

        map_anchor_embed = map_output["anchor_embed"]
        map_classification = map_output["classification"][-1].sigmoid()
        map_anchors = map_output["prediction"][-1]
        map_confidence = map_classification.max(dim=-1).values
        _, (map_instance_feature_selected, map_anchor_embed_selected) = topk(
            map_confidence, self.num_map, map_instance_feature, map_anchor_embed
        )

        # =========== get ego/temporal feature/anchor ===========
        bs, num_anchor, dim = instance_feature.shape
        (
            ego_feature,#torch.Size([6, 1, 256])
            ego_anchor,#[6, 1, 11]
            temp_instance_feature,#torch.Size([6, 901, 1, 256])
            temp_anchor,#torch.Size([6, 901, 1, 11])
            temp_mask,#torch.Size([6, 901, 1])
        ) = self.instance_queue.get(
            det_output,#dict_keys(['classification', 'prediction', 'quality', 'instance_feature', 'anchor_embed', 'instance_id'])
            feature_maps,#torch.Size([6, 89760, 256]) torch.Size([6, 4, 2]) torch.Size([6, 4])
            metas,#dict_keys(['img_metas', 'timestamp', 'projection_mat', 'image_wh', 'gt_depth', 'focal', 'gt_bboxes_3d', 'gt_labels_3d', 'gt_map_labels', 'gt_map_pts', 'gt_agent_fut_trajs', 'gt_agent_fut_masks', 'gt_ego_fut_trajs', 'gt_ego_fut_masks', 'gt_ego_fut_cmd', 'ego_status'])
            bs,#6
            mask,
            anchor_handler,
        )
        ego_anchor_embed = anchor_encoder(ego_anchor)#torch.Size([6, 1, 256])
        temp_anchor_embed = anchor_encoder(temp_anchor)#torch.Size([6, 901, 1, 256])
        temp_instance_feature = temp_instance_feature.flatten(0, 1)#torch.Size([5406, 1, 256])
        temp_anchor_embed = temp_anchor_embed.flatten(0, 1)#torch.Size([6, 901, 1, 256])
        temp_mask = temp_mask.flatten(0, 1)#torch.Size([6, 901, 1])

        # =========== mode anchor init ===========
        motion_anchor = self.get_motion_anchor(det_classification, det_anchors) #motion_anchor torch.Size([6, 900, 6, 12, 2]) #torch.Size([6, 900, 10])  torch.Size([6, 900, 11])
        plan_anchor = torch.tile( #torch.Size([6, 3, 6, 12, 2])
            self.plan_anchor[None], (bs, 1, 1, 1, 1)
        )

        # =========== mode query init ===========
        motion_mode_query = self.motion_anchor_encoder(gen_sineembed_for_position(motion_anchor[..., -1, :])) #torch.Size([6, 900, 6, 256])
        plan_pos = gen_sineembed_for_position(plan_anchor[..., -1, :]) #torch.Size([6, 3, 6, 256])
        plan_mode_query = self.plan_anchor_encoder(plan_pos).flatten(1, 2).unsqueeze(1)#torch.Size([6, 1, 18, 256])

        # =========== cat instance and ego ===========
        instance_feature_selected = torch.cat([instance_feature_selected, ego_feature], dim=1) #torch.Size([6, 50, 256]) torch.Size([6, 50, 256]) torch.Size([6, 1, 256])
        anchor_embed_selected = torch.cat([anchor_embed_selected, ego_anchor_embed], dim=1) #torch.Size([6, 50, 256]) torch.Size([6, 50, 256]) torch.Size([6, 1, 256])

        instance_feature = torch.cat([instance_feature, ego_feature], dim=1) #torch.Size([6, 900, 256]) torch.Size([6, 900, 256]) torch.Size([6, 1, 256])
        anchor_embed = torch.cat([anchor_embed, ego_anchor_embed], dim=1)#torch.Size([6, 900, 256]) torch.Size([6, 900, 256]) torch.Size([6, 1, 256])

        # =================== forward the layers ====================
        motion_classification = []
        motion_prediction = []
        planning_classification = []
        planning_classification_for_history = []
        planning_prediction = []
        planning_status = []
        planning_classification_refined = []
        planning_prediction_refined = []
        planning_status_refined = []
        c3a_phase_logits = []
        c3a_gate_records = []
        c3a_residual_records = []
        rcpc_outputs = []
        dense_risk_outputs = []
        selective_pareto_outputs = []
        smm_observability_decision = None
        (
            sample_tokens,
            prev_tokens,
            scene_tokens,
            causal_history_valid,
        ) = self._prepare_planning_history_for_batch(
            bs, plan_mode_query.device, plan_mode_query.dtype, metas
        )
        for i, op in enumerate(self.operation_order):
            #import pdb;pdb.set_trace()
            if self.layers[i] is None:
                continue
            elif op == "temp_gnn":
                #self.last_op = "temp_gnn"
                instance_feature = self.graph_model(#[5406, 1, 256])
                    i,
                    instance_feature.flatten(0, 1).unsqueeze(1),
                    temp_instance_feature,#torch.Size([5406, 1, 256])
                    temp_instance_feature,
                    query_pos=anchor_embed.flatten(0, 1).unsqueeze(1),#torch.Size([5406, 1, 256])
                    key_pos=temp_anchor_embed,#torch.Size([5406, 1, 256])
                    key_padding_mask=temp_mask,#torch.Size([5406, 1])
                )
                instance_feature = instance_feature.reshape(bs, num_anchor + 1, dim)#torch.Size([6, 901, 256])
            elif op == "gnn":
                #self.last_op = "gnn"
                instance_feature = self.graph_model(
                    i,
                    instance_feature,
                    instance_feature_selected,
                    instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=anchor_embed_selected,
                )
            elif op == "norm" or op == "ffn":
                #self.last_op = "norm"
                instance_feature = self.layers[i](instance_feature)
            elif op == "cross_gnn":
                #self.last_op = "cross_gnn"
                instance_feature = self.layers[i](
                    instance_feature,
                    key=map_instance_feature_selected,
                    query_pos=anchor_embed,
                    key_pos=map_anchor_embed_selected,
                )
            elif op == "refine":
                motion_query = motion_mode_query + (instance_feature + anchor_embed)[:, :num_anchor].unsqueeze(2)
                plan_query = plan_mode_query + (instance_feature + anchor_embed)[:, num_anchor:].unsqueeze(2)
                if self.causal_smm_adapter is not None:
                    smm_history_query = self.last_plan_query
                    smm_history_score = self.last_planning_classification
                    smm_history_valid_mask = causal_history_valid.unsqueeze(1)
                    smm_reliability_features = None
                    smm_reliability_availability = None
                    if self.use_causal_history_control and self.causal_history_len > 1:
                        smm_history_query = self._causal_smm_history_query
                        smm_history_score = self._causal_smm_history_score
                        smm_history_valid_mask = self._causal_smm_history_valid_mask
                        smm_reliability_features = self._causal_smm_reliability_features
                        smm_reliability_availability = (
                            self._causal_smm_reliability_availability
                        )
                    smm_count_before = int(
                        self.causal_smm_adapter.real_mamba_forward_call_count
                    )
                    plan_query = self.causal_smm_adapter(
                        current_query=plan_query,
                        history_query=smm_history_query,
                        history_score=smm_history_score,
                        history_valid_mask=smm_history_valid_mask,
                        reliability_features=smm_reliability_features,
                        reliability_availability=smm_reliability_availability,
                    )
                    smm_count_after = int(
                        self.causal_smm_adapter.real_mamba_forward_call_count
                    )
                    smm_observability_decision = {
                        "rank": int(self._get_causal_rank_id()),
                        "world_size": int(self._get_causal_world_size()),
                        "h1_cache_len_before": int(
                            len(self._causal_history_cache)
                            if self.causal_history_len == 1
                            else len(getattr(self._causal_sequence_history_cache, "_entries", {}))
                        ),
                        "batch_records": list(
                            self._latest_causal_history_batch_audit_records
                        ),
                        "smm_invocation_count_before": smm_count_before,
                        "smm_invocation_count_after": smm_count_after,
                    }
                #plan_query= self.sa_atten(plan_query, self.last_plan_query.to(plan_query.device))
                # import pdb;pdb.set_trace()

                (
                    motion_cls,
                    motion_reg,
                    plan_cls,#torch.Size([6, 1, 18])
                    plan_reg,#torch.Size([6, 1, 18, 12, 2])
                    plan_status,
                ) = self.layers[i](
                    motion_query, #([6, 900, 6, 256]
                    plan_query, #6, 1, 18, 256]
                    instance_feature[:, num_anchor:],
                    anchor_embed[:, num_anchor:],
                )
                if self.c3a_horizon_adapter is not None:
                    c3a_output = self.c3a_horizon_adapter(
                        plan_reg=plan_reg,
                        plan_query=plan_query,
                        h1_memory_query=self.last_plan_query,
                        h1_memory_valid=causal_history_valid,
                    )
                    plan_reg = c3a_output["refined_plan_reg"]
                    c3a_phase_logits.append(c3a_output["phase_logits"])
                    c3a_gate_records.append(c3a_output["gate"])
                    c3a_residual_records.append(c3a_output["residual_cumulative"])
                if self.dense_risk_scorer is not None:
                    dense_risk_output = self.dense_risk_scorer(
                        base_logits=plan_cls,
                        plan_reg=plan_reg,
                        plan_query=plan_query,
                        agent_features=instance_feature_selected,
                        map_features=map_instance_feature_selected,
                        last_final_planning=self.last_final_planning_prediction,
                        history_valid=causal_history_valid,
                        ego_status=metas.get("ego_status", None),
                    )
                    plan_cls = dense_risk_output["calibrated_logits"]
                    dense_risk_outputs.append(dense_risk_output)
                # A selective controller may change the executed mode, but it
                # must not rewrite the frozen T1 proposer's recurrent history.
                # Cache the exact pre-CSPC logits for the next-frame state.
                plan_cls_for_history = plan_cls
                if self.selective_pareto_controller is not None:
                    selective_output = self.selective_pareto_controller(
                        base_logits=plan_cls,
                        plan_reg=plan_reg,
                        plan_query=plan_query,
                        agent_features=instance_feature_selected,
                        map_features=map_instance_feature_selected,
                        last_final_planning=self.last_final_planning_prediction,
                        history_valid=causal_history_valid,
                        ego_status=metas.get("ego_status", None),
                    )
                    plan_cls = selective_output["calibrated_logits"]
                    selective_pareto_outputs.append(selective_output)
                if self.rcpc_adapter is not None:
                    rcpc_output = self.rcpc_adapter(
                        base_logits=plan_cls,
                        plan_reg=plan_reg,
                        plan_query=plan_query,
                        agent_features=instance_feature_selected,
                        map_features=map_instance_feature_selected,
                        last_final_planning=self.last_final_planning_prediction,
                        history_valid=causal_history_valid,
                        calibration_features=self._rcpc_calibration_features,
                        calibration_availability=(
                            self._rcpc_calibration_availability
                        ),
                        ego_status=metas.get("ego_status", None),
                    )
                    plan_cls = rcpc_output["calibrated_logits"]
                    plan_reg = rcpc_output["refined_plan_reg"]
                    rcpc_outputs.append(rcpc_output)

                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_classification_for_history.append(
                    plan_cls_for_history
                    if self.selective_pareto_controller is not None
                    else plan_cls
                )
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)

        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)
        #import pdb;pdb.set_trace()
        motion_output = {
            "classification": motion_classification, #[6, 900, 6]
            "prediction": motion_prediction,#[6, 900, 6, 12, 2])
            "period": self.instance_queue.period,#[6, 900]
            "anchor_queue": self.instance_queue.anchor_queue,#torch.Size([6, 900, 11])
        }

        #import pdb;pdb.set_trace()
        batch_size = plan_query.shape[0]
        # sample1_token = metas["img_metas"][1]["token"]
        # sample2_token = metas["img_metas"][2]["token"]
        # sample3_token = metas["img_metas"][3]["token"]
        # sample4_token = metas["img_metas"][4]["token"]
        # sample5_token = metas["img_metas"][5]["token"]
        # import pdb; pdb.set_trace()
        # if self.last_planning_classification==0:
        #     pass
        # else:
        #     pass

        #import pdb; pdb.set_trace()
        #判断是否是场景的第一个sample，如果是，则上一帧的结果应该是0（目的是防止不同场景的数据结合）
        # prev_sample_token = self.nusc.get('sample', sample0_token)["prev"]
        device = plan_query.device
        dtype = plan_query.dtype
        # # # fusion
        # device = plan_query.device
        # self.last_planning_classification=torch.zeros([batch_size, 1, 18]).to(device).detach()
        # self.last_planning_prediction=torch.zeros([batch_size, 1, 18, 12, 2]).to(device).detach()
        # self.last_plan_query = torch.zeros([batch_size, 1, 18, 256]).to(device).detach()
        # self.last_final_planning_prediction  = torch.zeros([batch_size, 12, 2]).to(device).detach()
        # self.last_ego_cmd = torch.zeros([batch_size, 3]).to(device).detach()



        enhanced_plan_query = plan_query
        last_final_planning_prediction = self.last_final_planning_prediction
        current_ego_cmd = metas['gt_ego_fut_cmd']
        # print(current_ego_cmd)
        # print(self.last_ego_cmd)
        # import pdb;pdb.set_trace()

        select_currect_best_planning_prediction, select_currect_idx, ego_mask = self.select_currect_best_planning(last_final_planning_prediction, planning_prediction, current_ego_cmd, self.last_ego_cmd)
        select_last_idx = self.select_former_best_query(self.last_plan_query, self.last_ego_cmd, self.last_planning_classification)
        last_query_mask = torch.zeros_like(self.last_plan_query)
        last_query_detach = self.last_plan_query.clone()
        batch_idx = torch.arange(batch_size, device=device)
        current_idx = torch.as_tensor(
            select_currect_idx, device=device, dtype=torch.long
        )
        last_query_detach[batch_idx, 0, current_idx, :] = (
            self.last_plan_query[batch_idx, 0, select_last_idx, :]
        )
        last_query_mask[batch_idx, 0, current_idx, :] = 1
        last_query_mask[ego_mask] = 0
        # import pdb; pdb.set_trace()
        history_mask = (
            self.last_plan_query.abs().sum(dim=(1, 2, 3), keepdim=True) > 0
        ).to(device=device, dtype=enhanced_plan_query.dtype)
        predicted_plan_query = self.next_token_predictor(
            self.last_plan_query,
            self.last_planning_classification,
            enhanced_plan_query,
        )
        enhanced_plan_query = (
            predicted_plan_query * history_mask
            + enhanced_plan_query * (1 - history_mask)
        )
        enhanced_plan_query = (
            enhanced_plan_query
            + last_query_detach * last_query_mask.to(enhanced_plan_query.device)
        )

        # refine 模块添加
        plan_cls_2th, plan_reg_2th, plan_status_2th = self.refine_2th_layer(enhanced_plan_query, instance_feature[:, num_anchor:], anchor_embed[:, num_anchor:])
        #(torch.Size([1, 1, 18]), torch.Size([1, 1, 18, 12, 2]), torch.Size([1, 1, 10]))
        planning_classification_refined.append(plan_cls_2th)
        planning_prediction_refined.append(plan_reg_2th)
        planning_status_refined.append(plan_status_2th)


        # 将当前帧的结果存到cache, 并进行detach
        last_final_planning_prediction = self.select_last_final_planning_prediction(
            planning_classification_for_history, planning_prediction, metas
        ).detach()
        self.last_planning_classification = planning_classification_for_history[
            0
        ].detach()
        self.last_planning_prediction = planning_prediction[0].detach()
        self.last_plan_query = plan_query.detach()
        if self.training and not self.use_causal_history_control:
            if self.forbid_gt_history_input:
                raise RuntimeError(
                    "forbid_gt_history_input=True blocks gt_ego_fut_trajs "
                    "from being written to planning history"
                )
            #import pdb;pdb.set_trace()
            self.last_final_planning_prediction = metas['gt_ego_fut_trajs'].detach()
        else:
            self.last_final_planning_prediction = last_final_planning_prediction.detach()
        self.last_ego_cmd = metas['gt_ego_fut_cmd'].detach()
        if self.use_causal_history_control:
            current_rank = self._get_causal_rank_id()
            next_slot_keys = []
            if self.causal_history_len > 1:
                for idx, token in enumerate(sample_tokens):
                    scene_token = scene_tokens[idx]
                    key = (scene_token, token)
                    sample_record = self.nusc.get("sample", token)
                    entry = self._make_causal_sequence_entry(
                        idx,
                        token,
                        sample_record["prev"],
                        scene_token,
                        sample_record,
                        metas,
                        current_rank,
                    )
                    self._causal_sequence_history_cache.write(entry)
                    self._causal_cache_write_order += 1
                    next_slot_keys.append(key)
                self._prune_causal_sequence_cache()
            else:
                next_cache = {}
                for idx, token in enumerate(sample_tokens):
                    scene_token = scene_tokens[idx]
                    key = (scene_token, token)
                    sample_record = self.nusc.get("sample", token)
                    cache_entry = {
                        "scene_token": scene_token,
                        "sample_token": token,
                        "timestamp": self._sample_record_timestamp_sec(
                            sample_record
                        ),
                        "rank_id": int(current_rank),
                        "last_planning_classification": self.last_planning_classification[
                            idx
                        ].detach().clone(),
                        "last_planning_prediction": self.last_planning_prediction[
                            idx
                        ].detach().clone(),
                        "last_plan_query": self.last_plan_query[idx].detach().clone(),
                        "last_final_planning_prediction": self.last_final_planning_prediction[
                            idx
                        ].detach().clone(),
                        "last_ego_cmd": self.last_ego_cmd[idx].detach().clone(),
                    }
                    if self.enable_rcpc:
                        translation, yaw, rotation = self._sample_record_pose(
                            sample_record
                        )
                        cache_entry.update(
                            {
                                "ego_translation": tuple(
                                    float(value) for value in translation
                                ),
                                "ego_yaw": float(yaw),
                                "ego2global_rotation": [
                                    float(value) for value in rotation
                                ],
                                "ego_status": self._ego_status_for_cache(
                                    metas, idx
                                ),
                            }
                        )
                    next_cache[key] = cache_entry
                    next_slot_keys.append(key)
                self._causal_history_cache = next_cache
            self._causal_last_local_slot_keys = next_slot_keys
        if smm_observability_decision is not None:
            h1_cache_len_after = int(
                len(self._causal_history_cache)
                if self.causal_history_len == 1
                else len(getattr(self._causal_sequence_history_cache, "_entries", {}))
            )
            self.smm_observability.record_decision(
                rank=smm_observability_decision["rank"],
                world_size=smm_observability_decision["world_size"],
                h1_cache_len_before=smm_observability_decision[
                    "h1_cache_len_before"
                ],
                h1_cache_len_after=h1_cache_len_after,
                batch_records=smm_observability_decision["batch_records"],
                smm_invocation_count_before=smm_observability_decision[
                    "smm_invocation_count_before"
                ],
                smm_invocation_count_after=smm_observability_decision[
                    "smm_invocation_count_after"
                ],
                global_update_context=os.environ.get(
                    "SMM_OBSERVABILITY_GLOBAL_UPDATE_CONTEXT",
                    "PENDING",
                ),
            )


        planning_output = {
            "classification": planning_classification,#[6, 1, 18] #3是command
            "prediction": planning_prediction,#[6, 1, 18, 12, 2]
            "status": planning_status,#[6, 1, 10]
            "classification_refined": planning_classification_refined,#[6, 1, 18] #3是command
            "prediction_refined": planning_prediction_refined,#[6, 1, 18, 12, 2]
            "status_refined": planning_status_refined,#[6, 1, 10]
            "c3a_phase_logits": c3a_phase_logits,
            "c3a_gate": c3a_gate_records,
            "c3a_residual_cumulative": c3a_residual_records,
            "c3a_enabled": bool(self.c3a_horizon_adapter is not None),
            "rcpc_outputs": rcpc_outputs,
            "rcpc_enabled": bool(self.rcpc_adapter is not None),
            "dense_risk_outputs": dense_risk_outputs,
            "dense_risk_enabled": bool(self.dense_risk_scorer is not None),
            "selective_pareto_outputs": selective_pareto_outputs,
            "selective_pareto_enabled": bool(
                self.selective_pareto_controller is not None
            ),
            "selective_history_isolation": bool(
                self.selective_history_isolation
            ),
            "rcpc_calibration_records": list(self._rcpc_calibration_records),
            "period": self.instance_queue.ego_period,
            "anchor_queue": self.instance_queue.ego_anchor_queue,
        }
        return motion_output, planning_output


    def loss(self,
        motion_model_outs,
        planning_model_outs,
        data,
        motion_loss_cache
    ):
        loss = {}
        motion_loss = self.loss_motion(motion_model_outs, data, motion_loss_cache)
        loss.update(motion_loss)
        planning_loss = self.loss_planning(planning_model_outs, data)
        loss.update(planning_loss)
        planning_loss_refined = self.loss_planning_refined(planning_model_outs, data)
        loss.update(planning_loss_refined)
        c3a_loss = self.loss_c3a_lppc(planning_model_outs, data)
        loss.update(c3a_loss)
        rcpc_loss = self.loss_rcpc(planning_model_outs, data)
        loss.update(rcpc_loss)
        dense_risk_loss = self.loss_dense_risk(planning_model_outs, data)
        loss.update(dense_risk_loss)
        selective_pareto_loss = self.loss_selective_pareto(
            planning_model_outs, data
        )
        loss.update(selective_pareto_loss)
        return loss

    @force_fp32(apply_to=("model_outs"))
    def loss_motion(self, model_outs, data, motion_loss_cache):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        output = {}
        for decoder_idx, (cls, reg) in enumerate(
            zip(cls_scores, reg_preds)
        ):
            (
                cls_target,
                cls_weight,
                reg_pred,
                reg_target,
                reg_weight,
                num_pos
            ) = self.motion_sampler.sample(
                reg,
                data["gt_agent_fut_trajs"],
                data["gt_agent_fut_masks"],
                motion_loss_cache,
            )
            num_pos = max(reduce_mean(num_pos), 1.0)

            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.motion_loss_cls(cls, cls_target, weight=cls_weight, avg_factor=num_pos)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)
            reg_pred = reg_pred.cumsum(dim=-2)
            reg_target = reg_target.cumsum(dim=-2)
            reg_loss = self.motion_loss_reg(
                reg_pred, reg_target, weight=reg_weight, avg_factor=num_pos
            )

            output.update(
                {
                    f"motion_loss_cls_{decoder_idx}": cls_loss,
                    f"motion_loss_reg_{decoder_idx}": reg_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_planning(self, model_outs, data):
        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        status_preds = model_outs["status"]
        if self.use_causal_history_control:
            self._causal_loss_gt_target_used = True
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target,
                cls_weight,
                reg_pred,
                reg_target,
                reg_weight,
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_{decoder_idx}": reg_loss,
                    f"planning_loss_status_{decoder_idx}": status_loss,
                }
            )

        return output

    def _masked_mean(self, value, mask):
        mask = mask.to(device=value.device, dtype=value.dtype)
        while mask.dim() < value.dim():
            mask = mask.unsqueeze(-1)
        denom = mask.expand_as(value).sum().clamp_min(1.0)
        return (value * mask).sum() / denom

    def _longitudinal_acceleration(self, increments):
        vy = increments[..., 1] / self.planning_dt
        ay = torch.zeros_like(vy)
        ay[..., 1:] = (vy[..., 1:] - vy[..., :-1]) / self.planning_dt
        return vy, ay

    @force_fp32(apply_to=("model_outs",))
    def loss_rcpc(self, model_outs, data):
        if self.rcpc_adapter is None:
            return {}
        if not bool(model_outs.get("rcpc_enabled", False)):
            raise RuntimeError("RCPC adapter exists but forward did not mark it enabled")
        rcpc_outputs = model_outs.get("rcpc_outputs", [])
        if not rcpc_outputs:
            raise RuntimeError("RCPC is enabled but forward produced no critic output")
        output = {}
        for decoder_idx, rcpc_output in enumerate(rcpc_outputs):
            decoder_loss = self.rcpc_adapter.loss(rcpc_output, data)
            for key, value in decoder_loss.items():
                output[f"{key}_{decoder_idx}"] = value
        return output

    @force_fp32(apply_to=("model_outs",))
    def loss_dense_risk(self, model_outs, data):
        if self.dense_risk_scorer is None:
            return {}
        if not bool(model_outs.get("dense_risk_enabled", False)):
            raise RuntimeError("CDRS exists but forward did not mark it enabled")
        dense_outputs = model_outs.get("dense_risk_outputs", [])
        if not dense_outputs:
            raise RuntimeError("CDRS is enabled but forward produced no output")
        output = {}
        for decoder_idx, dense_output in enumerate(dense_outputs):
            decoder_loss = self.dense_risk_scorer.loss(dense_output, data)
            for key, value in decoder_loss.items():
                output[f"{key}_{decoder_idx}"] = value
        return output

    @force_fp32(apply_to=("model_outs",))
    def loss_selective_pareto(self, model_outs, data):
        if self.selective_pareto_controller is None:
            return {}
        if not bool(model_outs.get("selective_pareto_enabled", False)):
            raise RuntimeError("CSPC exists but forward did not mark it enabled")
        selective_outputs = model_outs.get("selective_pareto_outputs", [])
        if not selective_outputs:
            raise RuntimeError("CSPC is enabled but forward produced no output")
        output = {}
        for decoder_idx, selective_output in enumerate(selective_outputs):
            decoder_loss = self.selective_pareto_controller.loss(
                selective_output, data
            )
            for key, value in decoder_loss.items():
                output[f"{key}_{decoder_idx}"] = value
        return output

    @force_fp32(apply_to=("model_outs"))
    def loss_c3a_lppc(self, model_outs, data):
        output = {}
        any_enabled = (
            self.enable_phase_loss
            or self.enable_far_loss
            or self.enable_longitudinal_loss
        )
        if not any_enabled:
            return output
        if not bool(model_outs.get("c3a_enabled", False)):
            raise RuntimeError("C3-A losses are enabled but horizon phase Mamba is disabled")

        phase_logits_list = model_outs.get("c3a_phase_logits", [])
        if self.enable_phase_loss:
            if not phase_logits_list:
                raise RuntimeError("enable_phase_loss=True but no c3a_phase_logits were produced")
            for decoder_idx, phase_logits in enumerate(phase_logits_list):
                phase_loss, phase_stats = phase_supervision_loss(
                    phase_logits,
                    data["gt_ego_fut_trajs"],
                    data["gt_ego_fut_masks"],
                    dt=self.planning_dt,
                    v_stop=self.phase_v_stop,
                    a_thr=self.phase_a_thr,
                )
                output[f"c3a_loss_phase_{decoder_idx}"] = (
                    phase_loss * self.lambda_phase
                )
                for key, value in phase_stats.items():
                    if isinstance(value, (int, float)):
                        output[f"c3a_phase_{key}_{decoder_idx}"] = phase_logits.new_tensor(
                            float(value)
                        )

        if not (self.enable_far_loss or self.enable_longitudinal_loss):
            return output

        cls_scores = model_outs["classification"]
        reg_preds = model_outs["prediction"]
        for decoder_idx, (cls, reg) in enumerate(zip(cls_scores, reg_preds)):
            (
                _sampled_cls,
                _cls_target,
                _cls_weight,
                reg_pred,
                reg_target,
                reg_weight,
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data["gt_ego_fut_trajs"],
                data["gt_ego_fut_masks"],
                data,
            )
            reg_pred_cum = reg_pred.cumsum(dim=-2)
            reg_target_cum = reg_target.cumsum(dim=-2)
            if self.enable_far_loss:
                far_slice = slice(7, 12)
                far_abs = (
                    reg_pred_cum[..., far_slice, :]
                    - reg_target_cum[..., far_slice, :]
                ).abs()
                far_mask = reg_weight[..., far_slice]
                far_loss = self._masked_mean(far_abs, far_mask)
                output[f"c3a_loss_far_{decoder_idx}"] = far_loss * self.lambda_far

            if self.enable_longitudinal_loss:
                y_loss = self._masked_mean(
                    (reg_pred_cum[..., 1] - reg_target_cum[..., 1]).abs(),
                    reg_weight,
                )
                pred_vy, pred_ay = self._longitudinal_acceleration(reg_pred)
                target_vy, target_ay = self._longitudinal_acceleration(reg_target)
                vy_loss = self._masked_mean((pred_vy - target_vy).abs(), reg_weight)
                ay_loss = self._masked_mean((pred_ay - target_ay).abs(), reg_weight)
                long_loss = 0.5 * y_loss + 0.3 * vy_loss + 0.2 * ay_loss
                output[f"c3a_loss_longitudinal_{decoder_idx}"] = (
                    long_loss * self.lambda_longitudinal
                )

        for key, value in list(output.items()):
            if torch.is_tensor(value) and not torch.isfinite(value).all():
                raise FloatingPointError(f"{key} contains NaN/Inf")
        return output

    def loss_planning_refined(self, model_outs, data):
        cls_scores = model_outs["classification_refined"]
        reg_preds = model_outs["prediction_refined"]
        status_preds = model_outs["status_refined"]
        if self.use_causal_history_control:
            self._causal_loss_gt_target_used = True
        output = {}
        for decoder_idx, (cls, reg, status) in enumerate(
            zip(cls_scores, reg_preds, status_preds)
        ):
            (
                cls,
                cls_target,
                cls_weight,
                reg_pred,
                reg_target,
                reg_weight,
            ) = self.planning_sampler.sample(
                cls,
                reg,
                data['gt_ego_fut_trajs'],
                data['gt_ego_fut_masks'],
                data,
            )
            cls = cls.flatten(end_dim=1)
            cls_target = cls_target.flatten(end_dim=1)
            cls_weight = cls_weight.flatten(end_dim=1)
            cls_loss = self.plan_loss_cls(cls, cls_target, weight=cls_weight)

            reg_weight = reg_weight.flatten(end_dim=1)
            reg_pred = reg_pred.flatten(end_dim=1)
            reg_target = reg_target.flatten(end_dim=1)
            reg_weight = reg_weight.unsqueeze(-1)

            reg_loss = self.plan_loss_reg(
                reg_pred, reg_target, weight=reg_weight
            )
            status_loss = self.plan_loss_status(status.squeeze(1), data['ego_status'])

            output.update(
                {
                    f"planning_loss_cls_refined_{decoder_idx}": cls_loss,
                    f"planning_loss_reg_refined_{decoder_idx}": reg_loss,
                    f"planning_loss_status_refined_{decoder_idx}": status_loss,
                }
            )

        return output

    @force_fp32(apply_to=("model_outs"))
    def post_process(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        motion_result = self.motion_decoder.decode(
            det_output["classification"],
            det_output["prediction"],
            det_output.get("instance_id"),
            det_output.get("quality"),
            motion_output,
        )
        planning_result = self.planning_decoder.decode(
            det_output,
            motion_output,
            planning_output,
            data,
        )

        return motion_result, planning_result
