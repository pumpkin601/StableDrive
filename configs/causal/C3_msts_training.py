import os

_base_ = "./C1_causal_smm_h1_real_mamba.py"

c3a_run_root = os.path.join(
    os.environ.get("STABLEDRIVE_ROOT", "."),
    "work_dirs/C3_msts_training",
)
work_dir = c3a_run_root

c3a_protocol_name = "GT-free self-history protocol under GT-command-conditioned planning decode"
c3a_experiment_name = "C3-A = C1 H1 Causal SMM + Longitudinal Phase-aware Horizon Mamba"

enable_horizon_phase_mamba = True
enable_phase_loss = True
enable_far_loss = True
enable_longitudinal_loss = True

phase_num_classes = 4
phase_names = [
    "stationary",
    "accelerating",
    "cruising",
    "decelerating",
]
longitudinal_axis = "y"
longitudinal_axis_index = 1
lateral_axis = "x"
lateral_axis_index = 0
far_horizon_start = 8
far_horizon_end = 12

lambda_phase = 0.05
lambda_far = 0.10
lambda_longitudinal = 0.10
phase_v_stop = 0.5
phase_a_thr = 0.5
planning_dt = 0.5

horizon_phase_mamba_cfg = dict(
    token_dims=256,
    phase_embed_dims=32,
    d_state=16,
    d_conv=4,
    expand=2,
    num_phase=phase_num_classes,
    require_real_mamba=True,
    allow_mamba_fallback=False,
)

model = dict(
    head=dict(
        motion_plan_head=dict(
            enable_horizon_phase_mamba=enable_horizon_phase_mamba,
            horizon_phase_mamba_cfg=horizon_phase_mamba_cfg,
            enable_phase_loss=enable_phase_loss,
            enable_far_loss=enable_far_loss,
            enable_longitudinal_loss=enable_longitudinal_loss,
            lambda_phase=lambda_phase,
            lambda_far=lambda_far,
            lambda_longitudinal=lambda_longitudinal,
            phase_v_stop=phase_v_stop,
            phase_a_thr=phase_a_thr,
            planning_dt=planning_dt,
        )
    )
)

c3a_semantic_spec = dict(
    protocol_name=c3a_protocol_name,
    phase_num_classes=phase_num_classes,
    phase_names=phase_names,
    longitudinal_axis=longitudinal_axis,
    longitudinal_axis_index=longitudinal_axis_index,
    lateral_axis=lateral_axis,
    lateral_axis_index=lateral_axis_index,
    planning_dt=planning_dt,
    far_horizon_start=far_horizon_start,
    far_horizon_end=far_horizon_end,
    forbidden_modules=dict(
        h4_history=False,
        svpc=False,
        candidate_score_calibration=False,
        global_18_candidate_calibration=False,
    ),
    unchanged_protocol=dict(
        gt_command_conditioned_decode=True,
        command_decode_mutation=False,
    ),
)

c3a_expected_motion_plan_head_overrides = dict(
        enable_horizon_phase_mamba=enable_horizon_phase_mamba,
        horizon_phase_mamba_cfg=horizon_phase_mamba_cfg,
        enable_phase_loss=enable_phase_loss,
        enable_far_loss=enable_far_loss,
        enable_longitudinal_loss=enable_longitudinal_loss,
        lambda_phase=lambda_phase,
        lambda_far=lambda_far,
        lambda_longitudinal=lambda_longitudinal,
        phase_v_stop=phase_v_stop,
        phase_a_thr=phase_a_thr,
        planning_dt=planning_dt,
)

c3a_disabled_equivalence_flags = dict(
    enable_horizon_phase_mamba=False,
    enable_phase_loss=False,
    enable_far_loss=False,
    enable_longitudinal_loss=False,
)
