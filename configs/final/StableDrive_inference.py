"""Deployable StableDrive: fixed T0/T1-off midpoint with MSTS retired."""

_base_ = "../causal/C1_causal_smm_h1_real_mamba.py"

load_from = "checkpoints/stabledrive_fixed_midpoint.pth"
resume_from = None
work_dir = "work_dirs/StableDrive_inference"

stabledrive_release = dict(
    alpha=0.5,
    checkpoint_operation="shared-weight arithmetic midpoint",
    inference_model="causal H1 SMM",
    msts_training_only=True,
    msts_runtime_enabled=False,
    ensemble=False,
    forward_passes=1,
)
