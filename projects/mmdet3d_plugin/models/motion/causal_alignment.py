from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np


DT_SECONDS = 0.5
PLANNING_HORIZON_STEPS = 12


def wrap_angle(angle: float) -> float:
    return float(math.atan2(math.sin(angle), math.cos(angle)))


def quaternion_yaw(rotation: Sequence[float]) -> float:
    """Return global yaw from a nuScenes quaternion [w, x, y, z]."""
    w, x, y, z = [float(v) for v in rotation]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def yaw_to_rot2d(yaw: float) -> np.ndarray:
    c = math.cos(float(yaw))
    s = math.sin(float(yaw))
    return np.array([[c, -s], [s, c]], dtype=np.float64)


@dataclass(frozen=True)
class EgoPose2D:
    translation_xy: np.ndarray
    yaw: float
    timestamp: float

    @property
    def rot(self) -> np.ndarray:
        return yaw_to_rot2d(self.yaw)

    @property
    def inv_rot(self) -> np.ndarray:
        return self.rot.T


def ego_pose_from_info(info: Dict) -> EgoPose2D:
    translation = np.asarray(info["ego2global_translation"], dtype=np.float64)
    timestamp = float(info["timestamp"])
    # Dataset infos store microseconds; dataloader metadata may already store
    # Unix seconds, which are also around 1e9. Only divide clear microsecond
    # timestamps to avoid double-scaling validation metadata.
    if timestamp > 1e12:
        timestamp = timestamp / 1e6
    return EgoPose2D(
        translation_xy=translation[:2].astype(np.float64),
        yaw=quaternion_yaw(info["ego2global_rotation"]),
        timestamp=timestamp,
    )


def transform_points_hist_to_current(
    points_hist: np.ndarray,
    hist_pose: EgoPose2D,
    current_pose: EgoPose2D,
) -> np.ndarray:
    points = np.asarray(points_hist, dtype=np.float64)
    flat = points.reshape(-1, 2)
    global_xy = flat @ hist_pose.rot.T + hist_pose.translation_xy[None, :]
    current_xy = (global_xy - current_pose.translation_xy[None, :]) @ current_pose.rot
    return current_xy.reshape(points.shape)


def transform_velocity_hist_to_current(
    velocity_hist: np.ndarray,
    hist_pose: EgoPose2D,
    current_pose: EgoPose2D,
) -> np.ndarray:
    velocity = np.asarray(velocity_hist, dtype=np.float64)
    flat = velocity.reshape(-1, 2)
    global_velocity = flat @ hist_pose.rot.T
    current_velocity = global_velocity @ current_pose.rot
    return current_velocity.reshape(velocity.shape)


def transform_heading_hist_to_current(
    heading_hist: float,
    hist_pose: EgoPose2D,
    current_pose: EgoPose2D,
) -> float:
    return wrap_angle(float(heading_hist) + hist_pose.yaw - current_pose.yaw)


def cumulative_from_increments(increments: np.ndarray) -> np.ndarray:
    return np.asarray(increments, dtype=np.float64).cumsum(axis=-2)


def is_cumulative_trajectory(traj: np.ndarray) -> bool:
    traj = np.asarray(traj)
    return traj.shape[-2:] == (PLANNING_HORIZON_STEPS, 2)


def current_index_for_history_age(age: int, dt: float = DT_SECONDS) -> int:
    if age < 1:
        raise ValueError("history age must be >= 1")
    # At 2Hz, t-1 is 0.5s old and current time corresponds to point 0.
    return int(age - 1)


def remaining_future_slice_for_age(age: int) -> Tuple[int, int]:
    start = current_index_for_history_age(age) + 1
    return start, PLANNING_HORIZON_STEPS


def observed_velocity_xy_from_ego_status(ego_status: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if ego_status is None:
        return None
    status = np.asarray(ego_status, dtype=np.float64)
    if status.shape[-1] < 9:
        return None
    # Converter stores velocity in status[6:9]. This codebase also uses index 6
    # as ego longitudinal speed (InstanceQueue writes it into VY), so map it to
    # planning xy as lateral=x=status[7], longitudinal=y=status[6].
    return np.array([float(status[7]), float(status[6])], dtype=np.float64)


def trajectory_velocity_at_index(traj: np.ndarray, index: int, dt: float = DT_SECONDS) -> np.ndarray:
    traj = np.asarray(traj, dtype=np.float64)
    if index < 0 or index >= traj.shape[-2]:
        raise ValueError(f"trajectory index {index} outside [0, {traj.shape[-2]})")
    if index == 0:
        prev = np.zeros(2, dtype=np.float64)
    else:
        prev = traj[index - 1]
    return (traj[index] - prev) / float(dt)


def trajectory_heading_at_index(traj: np.ndarray, index: int) -> float:
    velocity = trajectory_velocity_at_index(traj, index)
    if np.linalg.norm(velocity) < 1e-6:
        return math.pi / 2.0
    return float(math.atan2(velocity[1], velocity[0]))


def retrospective_calibration_features(
    historical_traj_cumulative: np.ndarray,
    age: int,
    hist_pose: EgoPose2D,
    current_pose: EgoPose2D,
    current_ego_status: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    traj = np.asarray(historical_traj_cumulative, dtype=np.float64)
    idx = current_index_for_history_age(age)
    available = idx < traj.shape[0]
    if not available:
        return {
            "available": False,
            "age": int(age),
            "current_index": int(idx),
            "reason": "history_horizon_too_short",
        }
    pred_current_hist = traj[idx]
    pred_current_aligned = transform_points_hist_to_current(
        pred_current_hist[None, :], hist_pose, current_pose
    )[0]
    hist_velocity = trajectory_velocity_at_index(traj, idx)
    velocity_aligned = transform_velocity_hist_to_current(
        hist_velocity[None, :], hist_pose, current_pose
    )[0]
    observed_velocity = observed_velocity_xy_from_ego_status(current_ego_status)
    heading_hist = trajectory_heading_at_index(traj, idx)
    heading_current = transform_heading_hist_to_current(heading_hist, hist_pose, current_pose)
    # In current ego coordinates, forward is +y, i.e. atan2(y, x) = pi/2.
    heading_error = wrap_angle(heading_current - math.pi / 2.0)
    e_vel = None
    if observed_velocity is not None:
        e_vel = float(np.linalg.norm(velocity_aligned - observed_velocity))
    return {
        "available": True,
        "age": int(age),
        "current_index": int(idx),
        "timestamp_gap_sec": float(current_pose.timestamp - hist_pose.timestamp),
        "pred_current_xy": [float(v) for v in pred_current_aligned.tolist()],
        "pred_current_velocity_xy": [float(v) for v in velocity_aligned.tolist()],
        "observed_velocity_xy": None
        if observed_velocity is None
        else [float(v) for v in observed_velocity.tolist()],
        "pred_current_heading_rad": float(heading_current),
        "e_pos": float(np.linalg.norm(pred_current_aligned)),
        "e_vel": e_vel,
        "e_yaw": float(abs(heading_error)),
        "e_progress": float(abs(pred_current_aligned[1])),
    }
