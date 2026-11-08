from typing import Optional

import numpy as np
import torch

from mmdet.core.bbox.builder import BBOX_CODERS

from projects.mmdet3d_plugin.core.box3d import *
from projects.mmdet3d_plugin.models.detection3d.decoder import *
from projects.mmdet3d_plugin.datasets.utils import box3d_to_corners


@BBOX_CODERS.register_module()
class SparseBox3DMotionDecoder(SparseBox3DDecoder):
    def __init__(self):
        super(SparseBox3DMotionDecoder, self).__init__()

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        motion_output=None,
        output_idx=-1,
    ):
        squeeze_cls = instance_id is not None

        cls_scores = cls_scores[output_idx].sigmoid()

        if squeeze_cls:
            cls_scores, cls_ids = cls_scores.max(dim=-1)
            cls_scores = cls_scores.unsqueeze(dim=-1)

        box_preds = box_preds[output_idx]
        bs, num_pred, num_cls = cls_scores.shape
        cls_scores, indices = cls_scores.flatten(start_dim=1).topk(
            self.num_output, dim=1, sorted=self.sorted
        )
        if not squeeze_cls:
            cls_ids = indices % num_cls
        if self.score_threshold is not None:
            mask = cls_scores >= self.score_threshold

        if quality[output_idx] is None:
            quality = None
        if quality is not None:
            centerness = quality[output_idx][..., CNS]
            centerness = torch.gather(centerness, 1, indices // num_cls)
            cls_scores_origin = cls_scores.clone()
            cls_scores *= centerness.sigmoid()
            cls_scores, idx = torch.sort(cls_scores, dim=1, descending=True)
            if not squeeze_cls:
                cls_ids = torch.gather(cls_ids, 1, idx)
            if self.score_threshold is not None:
                mask = torch.gather(mask, 1, idx)
            indices = torch.gather(indices, 1, idx)

        output = []
        anchor_queue = motion_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = motion_output["period"]

        for i in range(bs):
            category_ids = cls_ids[i]
            if squeeze_cls:
                category_ids = category_ids[indices[i]]
            scores = cls_scores[i]
            box = box_preds[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                category_ids = category_ids[mask[i]]
                scores = scores[mask[i]]
                box = box[mask[i]]
            if quality is not None:
                scores_origin = cls_scores_origin[i]
                if self.score_threshold is not None:
                    scores_origin = scores_origin[mask[i]]

            box = decode_box(box)
            trajs = motion_output["prediction"][-1]
            traj_cls = motion_output["classification"][-1].sigmoid()
            traj = trajs[i, indices[i] // num_cls]
            traj_cls = traj_cls[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                traj = traj[mask[i]]
                traj_cls = traj_cls[mask[i]]
            traj = traj.cumsum(dim=-2) + box[:, None, None, :2]
            output.append(
                {
                    "trajs_3d": traj.cpu(),
                    "trajs_score": traj_cls.cpu()
                }
            )

            temp_anchor = anchor_queue[i, indices[i] // num_cls]
            temp_period = period[i, indices[i] // num_cls]
            if self.score_threshold is not None:
                temp_anchor = temp_anchor[mask[i]]
                temp_period = temp_period[mask[i]]
            num_pred, queue_len = temp_anchor.shape[:2]
            temp_anchor = temp_anchor.flatten(0, 1)
            temp_anchor = decode_box(temp_anchor)
            temp_anchor = temp_anchor.reshape([num_pred, queue_len, box.shape[-1]])
            output[-1]['anchor_queue'] = temp_anchor.cpu()
            output[-1]['period'] = temp_period.cpu()

        return output


@BBOX_CODERS.register_module()
class HierarchicalPlanningDecoder(object):
    def __init__(
        self,
        ego_fut_ts,
        ego_fut_mode,
        use_rescore=False,
    ):
        super(HierarchicalPlanningDecoder, self).__init__()
        self.ego_fut_ts = ego_fut_ts
        self.ego_fut_mode = ego_fut_mode
        self.use_rescore = use_rescore

    def decode(
        self,
        det_output,
        motion_output,
        planning_output,
        data,
    ):
        classification = planning_output['classification'][-1]
        prediction = planning_output['prediction'][-1]
        bs = classification.shape[0]
        classification = classification.reshape(bs, 3, self.ego_fut_mode)
        prediction = prediction.reshape(bs, 3, self.ego_fut_mode, self.ego_fut_ts, 2).cumsum(dim=-2)
        classification, final_planning, rescore_collision = self.select(
            det_output, motion_output, classification, prediction, data
        )
        anchor_queue = planning_output["anchor_queue"]
        anchor_queue = torch.stack(anchor_queue, dim=2)
        period = planning_output["period"]
        rcpc_output = None
        rcpc_outputs = planning_output.get("rcpc_outputs", [])
        if rcpc_outputs:
            rcpc_output = rcpc_outputs[-1]
        selective_output = None
        selective_outputs = planning_output.get("selective_pareto_outputs", [])
        if selective_outputs:
            selective_output = selective_outputs[-1]
        command = data["gt_ego_fut_cmd"].argmax(dim=-1)
        output = []
        for i, (cls, pred) in enumerate(zip(classification, prediction)):
            record = {
                "planning_score": cls.sigmoid().cpu(),
                "planning": pred.cpu(),
                "final_planning": final_planning[i].cpu(),
                "ego_period": period[i].cpu(),
                "ego_anchor_queue": decode_box(anchor_queue[i]).cpu(),
            }
            if rescore_collision is not None:
                # The RCPC selector proposes before the legacy online
                # collision rescore.  Serialize that final safety-veto mask so
                # calibration can distinguish a proposal from an executed
                # takeover without reconstructing detector internals.
                record["planning_rescore_collision"] = (
                    rescore_collision[i].detach().cpu()
                )
            if rcpc_output is not None:
                cmd = int(command[i].item())
                if "risk_controller_applied" in rcpc_output:
                    displacement = rcpc_output["risk_candidate_displacement"][
                        i, cmd
                    ]
                    record.update(
                        {
                            "rcpc_risk_controller_applied": rcpc_output[
                                "risk_controller_applied"
                            ][i, cmd].detach().cpu(),
                            "rcpc_risk_predicted_pareto_safe": rcpc_output[
                                "risk_predicted_pareto_safe"
                            ][i, cmd].detach().cpu(),
                            "rcpc_risk_structural_safe": rcpc_output[
                                "risk_structural_safe"
                            ][i, cmd].detach().cpu(),
                            "rcpc_risk_metric_lcb": rcpc_output[
                                "risk_metric_lcb"
                            ][i, cmd].detach().cpu(),
                            "rcpc_risk_metric_median": rcpc_output[
                                "risk_metric_median"
                            ][i, cmd].detach().cpu(),
                            "rcpc_risk_analytic_improvement": rcpc_output[
                                "risk_analytic_improvement"
                            ][i, cmd].detach().cpu(),
                            "rcpc_risk_mean_displacement": displacement.mean()
                            .detach()
                            .cpu(),
                            "rcpc_risk_endpoint_displacement": displacement[-1]
                            .detach()
                            .cpu(),
                            "rcpc_base_winner": rcpc_output["base_winner"][
                                i, cmd
                            ].detach().cpu(),
                            "rcpc_critic_winner": rcpc_output["critic_winner"][
                                i, cmd
                            ].detach().cpu(),
                        }
                    )
                elif "trust_gate_applied" in rcpc_output:
                    record.update(
                        {
                            "rcpc_trust_gate_applied": rcpc_output[
                                "trust_gate_applied"
                            ][i, cmd].detach().cpu(),
                            "rcpc_trust_gate_probability": rcpc_output[
                                "trust_gate_probability"
                            ][i, cmd].detach().cpu(),
                            "rcpc_base_winner": rcpc_output["base_winner"][
                                i, cmd
                            ].detach().cpu(),
                            "rcpc_critic_winner": rcpc_output["critic_winner"][
                                i, cmd
                            ].detach().cpu(),
                        }
                    )
                if "selector_applied" in rcpc_output:
                    selector_displacement = rcpc_output[
                        "selector_candidate_displacement"
                    ][i, cmd]
                    selector_winner = rcpc_output["selector_winner"][i, cmd]
                    selected_displacement = selector_displacement[
                        selector_winner
                    ]
                    record.update(
                        {
                            "rcpc_selector_applied": rcpc_output[
                                "selector_applied"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_incumbent_winner": rcpc_output[
                                "selector_incumbent_winner"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_winner": selector_winner.detach().cpu(),
                            "rcpc_selector_metric_lcb": rcpc_output[
                                "selector_metric_lcb"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_metric_median": rcpc_output[
                                "selector_metric_median"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_predicted_pareto_safe": rcpc_output[
                                "selector_predicted_pareto_safe"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_structural_safe": rcpc_output[
                                "selector_structural_safe"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_observability_safe": rcpc_output[
                                "selector_observability_safe"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_eligible": rcpc_output[
                                "selector_eligible"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_analytic_improvement": rcpc_output[
                                "selector_analytic_improvement"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_candidate_utility": rcpc_output[
                                "selector_candidate_utility"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_innovation_gate": rcpc_output[
                                "selector_innovation_gate"
                            ][i, cmd].detach().cpu(),
                            "rcpc_selector_mean_displacement": (
                                selected_displacement.mean().detach().cpu()
                            ),
                            "rcpc_selector_endpoint_displacement": (
                                selected_displacement[-1].detach().cpu()
                            ),
                        }
                    )
                    if "selector_takeover_logit" in rcpc_output:
                        record.update(
                            {
                                "rcpc_selector_takeover_logit": rcpc_output[
                                    "selector_takeover_logit"
                                ][i, cmd].detach().cpu(),
                                "rcpc_selector_takeover_probability": rcpc_output[
                                    "selector_takeover_probability"
                                ][i, cmd].detach().cpu(),
                                "rcpc_selector_takeover_proposal_score": rcpc_output[
                                    "selector_takeover_proposal_score"
                                ][i, cmd].detach().cpu(),
                                "rcpc_selector_direct_eligible": rcpc_output[
                                    "selector_direct_eligible"
                                ][i, cmd].detach().cpu(),
                                "rcpc_selector_direct_proposed_winner": rcpc_output[
                                    "selector_direct_proposed_winner"
                                ][i, cmd].detach().cpu(),
                                "rcpc_selector_proposed_winner": rcpc_output[
                                    "selector_proposed_winner"
                                ][i, cmd].detach().cpu(),
                            }
                        )
                    if "selector_margin_prediction" in rcpc_output:
                        record.update(
                            {
                                "rcpc_selector_margin_prediction": rcpc_output[
                                    "selector_margin_prediction"
                                ][i, cmd].detach().cpu(),
                                "rcpc_selector_margin_joint_logit": rcpc_output[
                                    "selector_margin_joint_logit"
                                ][i, cmd].detach().cpu(),
                                "rcpc_selector_margin_evidence": rcpc_output[
                                    "selector_margin_evidence"
                                ][i, cmd].detach().cpu(),
                                "rcpc_selector_margin_safety_score": rcpc_output[
                                    "selector_margin_safety_score"
                                ][i, cmd].detach().cpu(),
                                "rcpc_selector_margin_probability": rcpc_output[
                                    "selector_margin_probability"
                                ][i, cmd].detach().cpu(),
                            }
                        )
            if selective_output is not None:
                cmd = int(command[i].item())
                metric = selective_output["score_metrics"].reshape(
                    bs, 3, self.ego_fut_mode, -1
                )
                base_logits = selective_output["base_logits"].reshape(
                    bs, 3, self.ego_fut_mode
                )
                calibrated_logits = selective_output["calibrated_logits"].reshape(
                    bs, 3, self.ego_fut_mode
                )
                record.update(
                    {
                        "cspc_base_mode": selective_output["base_mode"][
                            i, cmd
                        ].detach().cpu(),
                        "cspc_takeover_mode": selective_output["takeover_mode"][
                            i, cmd
                        ].detach().cpu(),
                        "cspc_takeover_applied": selective_output[
                            "takeover_applied"
                        ][i, cmd].detach().cpu(),
                        "cspc_takeover_logits": selective_output[
                            "takeover_logits"
                        ][i, cmd].detach().cpu(),
                        "cspc_takeover_probability": selective_output[
                            "takeover_probability"
                        ][i, cmd].detach().cpu(),
                        "cspc_takeover_eligible": selective_output[
                            "takeover_eligible"
                        ][i, cmd].detach().cpu(),
                        "cspc_predicted_guard_all": selective_output[
                            "predicted_guard_all"
                        ][i, cmd].detach().cpu(),
                        "cspc_predicted_weighted_ratio": selective_output[
                            "predicted_weighted_ratio"
                        ][i, cmd].detach().cpu(),
                        "cspc_predicted_metrics": metric[i, cmd].detach().cpu(),
                        "cspc_base_logits": base_logits[i, cmd].detach().cpu(),
                        "cspc_calibrated_logits": calibrated_logits[
                            i, cmd
                        ].detach().cpu(),
                    }
                )
            output.append(record)

        return output

    def select(
        self,
        det_output,
        motion_output,
        plan_cls,
        plan_reg,
        data,
    ):
        det_classification = det_output["classification"][-1].sigmoid()
        det_anchors = det_output["prediction"][-1]
        det_confidence = det_classification.max(dim=-1).values
        motion_cls = motion_output["classification"][-1].sigmoid()
        motion_reg = motion_output["prediction"][-1]

        # cmd select
        bs = motion_cls.shape[0]
        bs_indices = torch.arange(bs, device=motion_cls.device)
        cmd = data['gt_ego_fut_cmd'].argmax(dim=-1)
        plan_cls_full = plan_cls.detach().clone()
        plan_cls = plan_cls[bs_indices, cmd]
        plan_reg = plan_reg[bs_indices, cmd]

        # rescore
        rescore_collision = None
        if self.use_rescore:
            plan_cls, rescore_collision = self.rescore(
                plan_cls,
                plan_reg,
                motion_cls,
                motion_reg,
                det_anchors,
                det_confidence,
            )
        plan_cls_full[bs_indices, cmd] = plan_cls
        mode_idx = plan_cls.argmax(dim=-1)
        final_planning = plan_reg[bs_indices, mode_idx]
        return plan_cls_full, final_planning, rescore_collision

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

        # ``ego_box`` has already been flattened to [N, 7].  Apply the
        # UniAD-style longitudinal center offset to every box coordinate;
        # indexing the first dimension here would corrupt only rows 0 and 1.
        ego_box[..., X] += offset * torch.cos(ego_box[..., YAW])
        ego_box[..., Y] += offset * torch.sin(ego_box[..., YAW])
        col = check_collision(ego_box, motion_box)
        col = col.reshape(bs, num_anchor, num_motion_mode, num_ego_mode, ts).permute(0, 3, 1, 2, 4)
        col = col.flatten(2, -1).any(dim=-1)
        all_col = col.all(dim=-1)
        col[all_col] = False # for case that all modes collide, no need to rescore
        score_offset = col.float() * -999
        plan_cls = plan_cls + score_offset
        return plan_cls, col


def check_collision(boxes1, boxes2):
    '''
        A rough check for collision detection:
            check if any corner point of boxes1 is inside boxes2 and vice versa.

        boxes1: tensor with shape [N, 7], [x, y, z, w, l, h, yaw]
        boxes2: tensor with shape [N, 7]
    '''
    col_1 = corners_in_box(boxes1.clone(), boxes2.clone())
    col_2 = corners_in_box(boxes2.clone(), boxes1.clone())
    collision = torch.logical_or(col_1, col_2)

    return collision

def corners_in_box(boxes1, boxes2):
    if  boxes1.shape[0] == 0 or boxes2.shape[0] == 0:
        return False

    boxes1_yaw = boxes1[:, 6].clone()
    boxes1_loc = boxes1[:, :3].clone()
    cos_yaw = torch.cos(-boxes1_yaw)
    sin_yaw = torch.sin(-boxes1_yaw)
    rot_mat_T = torch.stack(
        [
            torch.stack([cos_yaw, sin_yaw]),
            torch.stack([-sin_yaw, cos_yaw]),
        ]
    )
    # translate and rotate boxes
    boxes1[:, :3] = boxes1[:, :3] - boxes1_loc
    boxes1[:, :2] = torch.einsum('ij,jki->ik', boxes1[:, :2], rot_mat_T)
    boxes1[:, 6] = boxes1[:, 6] - boxes1_yaw

    boxes2[:, :3] = boxes2[:, :3] - boxes1_loc
    boxes2[:, :2] = torch.einsum('ij,jki->ik', boxes2[:, :2], rot_mat_T)
    boxes2[:, 6] = boxes2[:, 6] - boxes1_yaw

    corners_box2 = box3d_to_corners(boxes2)[:, [0, 3, 7, 4], :2]
    corners_box2 = torch.from_numpy(corners_box2).to(boxes2.device)
    H = boxes1[:, [3]]
    W = boxes1[:, [4]]

    collision = torch.logical_and(
        torch.logical_and(corners_box2[..., 0] <= H / 2, corners_box2[..., 0] >= -H / 2),
        torch.logical_and(corners_box2[..., 1] <= W / 2, corners_box2[..., 1] >= -W / 2),
    )
    collision = collision.any(dim=-1)

    return collision
