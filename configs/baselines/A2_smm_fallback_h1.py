_base_ = ["../_base_/mamba_mpi_6s.py"]

# Fallback / no-real-Mamba ablation config.
# Purpose:
# - keep the same Mamba-MPI insertion point, alpha, residual path, and history input
# - disable official real Mamba sequence modeling
# - keep the main training protocol aligned with the real_mamba full-train run

work_dir = None
load_from = "checkpoints/external/sparsedrive_stage2_official_v1.0.pth"
resume_from = None

total_batch_size = 24
num_gpus = 4
batch_size = total_batch_size // num_gpus
num_iters_per_epoch = 1172
num_epochs = 10
checkpoint_epoch_interval = 2

checkpoint_config = dict(interval=2344)
log_config = dict(
    interval=20,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        dict(type="TensorboardLoggerHook"),
    ],
)

model = dict(
    head=dict(
        motion_plan_head=dict(
            smm_cfg=dict(
                enable=True,
                embed_dims=256,
                num_modes=18,
                history_len=1,
                use_real_mamba=False,
                fallback_if_unavailable=True,
            )
        )
    )
)
