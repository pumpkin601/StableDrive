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
from .next_token_prediction import NextTokenPredictor
from .selective_momentum_memory import SelectiveMomentumMemory
@HEADS.register_module()
class MambaMotionPlanningHead6s(BaseModule):
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
        use_rescore= True,
        smm_cfg=None,

    ):
        super(MambaMotionPlanningHead6s, self).__init__()
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
        self.smm_history_plan_query = torch.zeros(0, 1, 18, embed_dims)
        self.smm_history_planning_classification = torch.zeros(0, 1, 18)
        self.smm_history_valid_mask = torch.zeros(0, 1, dtype=torch.bool)
        self.last_ego_cmd = torch.zeros(0, 3)
        self.last_sample_tokens = []
        self.nusc = NuScenes(version='v1.0-trainval', dataroot="data/nuscenes/", verbose=True)
        self.use_rescore = use_rescore
        self.debug_enabled = os.environ.get("MAMBA_MPI_DEBUG", "0").lower() not in {
            "",
            "0",
            "false",
            "no",
        }
        self.debug_max_steps = int(os.environ.get("MAMBA_MPI_DEBUG_MAX_STEPS", "2"))
        self._debug_forward_idx = 0
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
        self.smm = None
        self.smm_cfg = copy.deepcopy(smm_cfg)
        if self.smm_cfg is not None and self.smm_cfg.get("enable", False):
            smm_cfg = copy.deepcopy(self.smm_cfg)
            smm_cfg.pop("enable", None)
            self.smm = SelectiveMomentumMemory(**smm_cfg)

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

    def _debug_print(self, message):
        if not self.debug_enabled:
            return
        if self._debug_forward_idx > self.debug_max_steps:
            return
        print(f"[MAMBA_MPI_DEBUG][HEAD][step={self._debug_forward_idx}] {message}")

    def _smm_history_len(self):
        if self.smm is None:
            return 1
        return max(int(getattr(self.smm, "history_len", 1)), 1)

    def _reset_temporal_cache_at(self, reset_mask):
        if not bool(reset_mask.any().item()):
            return
        self.last_planning_classification[reset_mask] = 0
        self.last_planning_prediction[reset_mask] = 0
        self.last_plan_query[reset_mask] = 0
        self.smm_history_plan_query[reset_mask] = 0
        self.smm_history_planning_classification[reset_mask] = 0
        self.smm_history_valid_mask[reset_mask] = False
        self.last_final_planning_prediction[reset_mask] = 0
        self.last_ego_cmd[reset_mask] = 0

    def init_weights(self):
        for i, op in enumerate(self.operation_order):
            if self.layers[i] is None:
                continue
            elif op != "refine":
                for p in self.layers[i].parameters():
                    if p.dim() > 1:
                        nn.init.xavier_uniform_(p)

        for m in self.modules():
            if hasattr(m, "init_weight"):
                m.init_weight()

    def _prepare_temporal_cache(self, batch_size, device, dtype, metas):
        smm_history_len = self._smm_history_len()
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
            self.smm_history_plan_query = torch.zeros(
                batch_size, smm_history_len, 18, self.embed_dims,
                device=device, dtype=dtype
            )
            self.smm_history_planning_classification = torch.zeros(
                batch_size, smm_history_len, 18, device=device, dtype=dtype
            )
            self.smm_history_valid_mask = torch.zeros(
                batch_size, smm_history_len, device=device, dtype=torch.bool
            )
            self.last_final_planning_prediction = torch.zeros(
                batch_size, self.ego_fut_ts, 2, device=device, dtype=dtype
            )
            self.last_ego_cmd = torch.zeros(
                batch_size, 3, device=device, dtype=dtype
            )
            self.last_sample_tokens = [None for _ in range(batch_size)]
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
            if (
                self.smm_history_plan_query.shape[0] != batch_size
                or self.smm_history_plan_query.shape[1] != smm_history_len
            ):
                self.smm_history_plan_query = torch.zeros(
                    batch_size, smm_history_len, 18, self.embed_dims,
                    device=device, dtype=dtype
                )
                self.smm_history_planning_classification = torch.zeros(
                    batch_size, smm_history_len, 18, device=device, dtype=dtype
                )
                self.smm_history_valid_mask = torch.zeros(
                    batch_size, smm_history_len, device=device, dtype=torch.bool
                )
            else:
                self.smm_history_plan_query = self.smm_history_plan_query.to(
                    device=device, dtype=dtype
                ).detach()
                self.smm_history_planning_classification = (
                    self.smm_history_planning_classification.to(
                        device=device, dtype=dtype
                    ).detach()
                )
                self.smm_history_valid_mask = self.smm_history_valid_mask.to(
                    device=device, dtype=torch.bool
                ).detach()
            self.last_final_planning_prediction = (
                self.last_final_planning_prediction.to(
                    device=device, dtype=dtype
                ).detach()
            )
            self.last_ego_cmd = self.last_ego_cmd.to(
                device=device, dtype=dtype
            ).detach()

        reset_flags = []
        for batch_idx, item in enumerate(metas["img_metas"]):
            sample = self.nusc.get("sample", item["token"])
            prev_token = sample["prev"]
            cached_token = self.last_sample_tokens[batch_idx]
            reset_flags.append(prev_token == "" or prev_token != cached_token)
        reset_mask = torch.tensor(reset_flags, device=device, dtype=torch.bool)
        self._reset_temporal_cache_at(reset_mask)

    def _append_smm_history(self, plan_query, planning_classification):
        if self.smm is None:
            return
        latest_query = plan_query.detach()
        latest_score = planning_classification.detach()
        latest_valid = torch.ones(
            latest_query.shape[0], 1, device=latest_query.device, dtype=torch.bool
        )
        if self.smm_history_plan_query.shape[1] == 1:
            self.smm_history_plan_query = latest_query
            self.smm_history_planning_classification = latest_score
            self.smm_history_valid_mask = latest_valid
            return
        self.smm_history_plan_query = torch.cat(
            [self.smm_history_plan_query[:, 1:].detach(), latest_query], dim=1
        )
        self.smm_history_planning_classification = torch.cat(
            [
                self.smm_history_planning_classification[:, 1:].detach(),
                latest_score,
            ],
            dim=1,
        )
        self.smm_history_valid_mask = torch.cat(
            [self.smm_history_valid_mask[:, 1:].detach(), latest_valid], dim=1
        )

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
        plan_query = self.sa_atten_layer(torch.cat([plan_query,last_query],dim=-1).permute(0,3,1,2))
        return plan_query.permute(0,2,3,1)
    def select_last_final_planning_prediction(self,planning_classification,planning_prediction,metas):
        #print("hhhhhhhhhhhhhhhhhhhhhh",len(planning_classification))
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
        ego_mask=(current_ego_cmd.argmax(-1) == last_ego_cmd.argmax(-1))
        distances_euclidean = self.euclidean_distance(last_final_planning_prediction.unsqueeze(1).repeat(1,18,1,1), planning_prediction[0].squeeze(1))
        min_idx = torch.argmin(distances_euclidean, dim=-1).cpu().numpy().tolist()
        bs = planning_prediction[0].squeeze(1).shape[0]
        select_currect_best_planning_prediction = torch.where(ego_mask.view(-1, 1, 1), planning_prediction[0].squeeze(1)[torch.arange(bs), min_idx], torch.full((12, 2), 999.9).repeat(bs, 1, 1).to(ego_mask.device))
        return select_currect_best_planning_prediction, min_idx, ego_mask

    def select_former_best_query(self, plan_query, last_ego_cmd, planning_classification):
        last_ego_cmd = last_ego_cmd.repeat_interleave(6, dim=1)
        planning_classification = planning_classification.squeeze(1).sigmoid() * last_ego_cmd

        return planning_classification.argmax(-1)

    def euclidean_distance(self, TA, TB):
        TA = TA.cumsum(dim=-2)
        TB = TB.cumsum(dim=-2)
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
        self._debug_forward_idx += 1

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
        planning_prediction = []
        planning_status = []
        planning_classification_refined = []
        planning_prediction_refined = []
        planning_status_refined = []
        self._prepare_temporal_cache(bs, instance_feature.device, instance_feature.dtype, metas)
        for i, op in enumerate(self.operation_order):
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
                self._debug_print(
                    f"refine start plan_query_shape={tuple(plan_query.shape)} "
                    f"plan_query_requires_grad={plan_query.requires_grad} "
                    f"history_nonzero={bool((self.last_plan_query.abs().sum() > 0).item())}"
                )
                if self.smm is not None:
                    plan_query = self.smm(
                        current_query=plan_query,
                        history_query=self.smm_history_plan_query,
                        history_score=self.smm_history_planning_classification,
                        history_valid_mask=self.smm_history_valid_mask,
                    )
                    self._debug_print(
                        f"after smm plan_query_shape={tuple(plan_query.shape)} "
                        f"plan_query_requires_grad={plan_query.requires_grad} "
                        f"plan_query_finite={bool(torch.isfinite(plan_query).all().item())}"
                    )
                #plan_query= self.sa_atten(plan_query, self.last_plan_query.to(plan_query.device))

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

                motion_classification.append(motion_cls)
                motion_prediction.append(motion_reg)
                planning_classification.append(plan_cls)
                planning_prediction.append(plan_reg)
                planning_status.append(plan_status)

        self.instance_queue.cache_motion(instance_feature[:, :num_anchor], det_output, metas)
        self.instance_queue.cache_planning(instance_feature[:, num_anchor:], plan_status)
        motion_output = {
            "classification": motion_classification, #[6, 900, 6]
            "prediction": motion_prediction,#[6, 900, 6, 12, 2])
            "period": self.instance_queue.period,#[6, 900]
            "anchor_queue": self.instance_queue.anchor_queue,#torch.Size([6, 900, 11])
        }

        batch_size = plan_query.shape[0]
        # sample1_token = metas["img_metas"][1]["token"]
        # sample2_token = metas["img_metas"][2]["token"]
        # sample3_token = metas["img_metas"][3]["token"]
        # sample4_token = metas["img_metas"][4]["token"]
        # sample5_token = metas["img_metas"][5]["token"]
        # if self.last_planning_classification==0:
        #     pass
        # else:
        #     pass

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
        self._debug_print(
            f"enhanced_plan_query_shape={tuple(enhanced_plan_query.shape)} "
            f"enhanced_plan_query_requires_grad={enhanced_plan_query.requires_grad} "
            f"enhanced_plan_query_finite={bool(torch.isfinite(enhanced_plan_query).all().item())}"
        )

        # refine 模块添加
        plan_cls_2th, plan_reg_2th, plan_status_2th = self.refine_2th_layer(enhanced_plan_query, instance_feature[:, num_anchor:], anchor_embed[:, num_anchor:])
        #(torch.Size([1, 1, 18]), torch.Size([1, 1, 18, 12, 2]), torch.Size([1, 1, 10]))
        planning_classification_refined.append(plan_cls_2th)
        planning_prediction_refined.append(plan_reg_2th)
        planning_status_refined.append(plan_status_2th)


        # 将当前帧的结果存到cache, 并进行detach
        last_final_planning_prediction =self.select_last_final_planning_prediction(planning_classification,planning_prediction,metas).detach()
        self.last_planning_classification = planning_classification[0].detach()
        self.last_planning_prediction = planning_prediction[0].detach()
        self.last_plan_query = plan_query.detach()
        self._append_smm_history(plan_query, planning_classification[0])
        if self.training:
            self.last_final_planning_prediction = metas['gt_ego_fut_trajs'].detach()
        else:
            self.last_final_planning_prediction = last_final_planning_prediction.detach()
        self.last_ego_cmd = metas['gt_ego_fut_cmd'].detach()
        self.last_sample_tokens = [
            item["token"] for item in metas["img_metas"]
        ]


        planning_output = {
            "classification": planning_classification,#[6, 1, 18] #3是command
            "prediction": planning_prediction,#[6, 1, 18, 12, 2]
            "status": planning_status,#[6, 1, 10]
            "classification_refined": planning_classification_refined,#[6, 1, 18] #3是command
            "prediction_refined": planning_prediction_refined,#[6, 1, 18, 12, 2]
            "status_refined": planning_status_refined,#[6, 1, 10]
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

    def loss_planning_refined(self, model_outs, data):
        cls_scores = model_outs["classification_refined"]
        reg_preds = model_outs["prediction_refined"]
        status_preds = model_outs["status_refined"]
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
