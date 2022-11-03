from typing import Optional, Tuple, Union

from cv2 import SOLVEPNP_AP3P, Rodrigues, solvePnPRansac, solvePnPRefineLM
import numpy as np
from torch import Tensor

from .typing import TensorOrArray, TensorOrArrayOrList


def distort(k: float, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """COLMAP - src/base/camera_models.h:747"""
    u2 = u * u
    v2 = v * v
    r2 = u2 + v2
    radial = k * r2
    du = radial * u
    dv = radial * v
    return du, dv


def undistort(k: float, u: np.ndarray, v: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dtype = u.dtype
    u, v = u.astype(np.float64), v.astype(np.float64)

    NUM_ITERATIONS = 100
    MAX_STEP_NORM = 1e-10
    REL_STEP_SIZE = 1e-6
    DOUBLE_EPS = np.finfo(np.float64).eps

    u0 = np.copy(u)
    v0 = np.copy(v)

    for _ in range(NUM_ITERATIONS):
        step0 = np.maximum(DOUBLE_EPS, np.abs(REL_STEP_SIZE * u))
        step1 = np.maximum(DOUBLE_EPS, np.abs(REL_STEP_SIZE * v))
        du, dv = distort(k, u, v)
        du_0b, dv_0b = distort(k, u - step0, v)
        du_0f, dv_0f = distort(k, u + step0, v)
        du_1b, dv_1b = distort(k, u, v - step1)
        du_1f, dv_1f = distort(k, u, v + step1)

        # fmt: off
        J = np.stack([
            np.stack([
                1 + (du_0f - du_0b) / (2 * step0),
                (du_1f - du_1b) / (2 * step1),
            ], axis=-1),
            np.stack([
                (dv_0f - dv_0b) / (2 * step0),
                1 + (dv_1f - dv_1b) / (2 * step1),
            ], axis=-1),
        ], axis=1)
        # fmt: on

        x = np.stack(
            [
                u + du - u0,
                v + dv - v0,
            ],
            axis=-1,
        )[..., None]
        step_u, step_v = np.linalg.solve(J, x).transpose(1, 0, 2).squeeze(-1)

        u -= step_u
        v -= step_v

        if np.max(step_u * step_u + step_v * step_v) < MAX_STEP_NORM:
            break

    u, v = u.astype(dtype), v.astype(dtype)
    return u, v


def get_numpy(data: TensorOrArrayOrList) -> np.ndarray:
    if isinstance(data, Tensor):
        out = data.cpu().data.numpy()
    elif isinstance(data, list):
        out = np.array(data)
    else:
        out = data

    assert isinstance(out, np.ndarray)
    return out


def project3d_normalized(
    R: TensorOrArray,
    t: TensorOrArray,
    pts3d: TensorOrArray,
    radial: Optional[float] = None,
    return_valid: bool = False,
) -> Union[TensorOrArray, Tuple[TensorOrArray, TensorOrArray]]:
    # Move points to camera space
    pts3d_cam = pts3d @ R.T + t

    # Bring it to the normalized image plane at z=1
    pts3d_norm = pts3d_cam / pts3d_cam[:, -1, None]

    # Distort if needed
    if radial is not None:
        assert isinstance(
            pts3d_norm, np.ndarray
        ), "Unable to apply radial distortion with torch.Tensor arguments"
        du, dv = distort(radial, pts3d_norm[:, 0], pts3d_norm[:, 1])
        pts3d_norm = pts3d_norm + np.stack([du, dv, np.zeros_like(dv)], axis=1)

    if return_valid:
        # only consider points in front of the camera
        valid = pts3d_cam[:, -1] >= 0
        return pts3d_norm[:, :2], valid
    else:
        return pts3d_norm[:, :2]


def project_points3d(
    K: TensorOrArrayOrList,
    R: TensorOrArrayOrList,
    t: TensorOrArrayOrList,
    pts3d: TensorOrArrayOrList,
    radial: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3D points to 2D points using extrinsics and intrinsics
    Args:
        - K: camera intrinc matrix (3, 3)
        - R: world to camera rotation (3, 3)
        - t: world to camera translation (3,)
        - pts3d: 3D points (N, 3)
        - radial: single distortion coefficient. It presents the coefficient k1 in https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html#ga7dfb72c9cf9780a347fbe3d1c47e5d5a
    Return:
        - pts2d: projected 2D points (N, 2)
        - valid: bool mask, indicates the 3d points that are projected in front of the camera
    """

    K = get_numpy(K)
    R = get_numpy(R)
    t = get_numpy(t)
    pts3d = get_numpy(pts3d)

    pts2d_norm, valid = project3d_normalized(
        R, t, pts3d, radial=radial, return_valid=True
    )
    assert isinstance(pts2d_norm, np.ndarray)
    assert isinstance(valid, np.ndarray)
    pts3d_norm = np.concatenate([pts2d_norm, np.ones((len(pts2d_norm), 1))], axis=1)

    # Transform to pixel space. Last column is already guaranteed to be set to 1
    pixels = pts3d_norm @ K.T
    return pixels[:, :2], valid


def points2d_to_bearing_vector(
    pts2d: np.ndarray, K: np.ndarray, vec_dim: int = 2, radial: Optional[float] = None
) -> np.ndarray:
    pts2d_homo = np.concatenate([pts2d, np.ones((len(pts2d), 1))], axis=-1)
    bvecs = np.linalg.solve(K, pts2d_homo.T)

    if radial is not None:
        bvecs[:2] = np.stack(undistort(radial, bvecs[0], bvecs[1]))
    bvecs = bvecs[:vec_dim].T
    return bvecs.astype(pts2d.dtype)


def estimate_pose(
    pts2d_bvs: TensorOrArray,
    pts3d: TensorOrArray,
    ransac_thres: float = 0.001,
    iterations_count: int = 1000,
    confidence: float = 0.99,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    if isinstance(pts2d_bvs, Tensor):
        pts2d_bvs = pts2d_bvs.cpu().data.numpy()
    if isinstance(pts3d, Tensor):
        pts3d = pts3d.cpu().data.numpy()

    if len(pts2d_bvs) < 4:
        return None

    # Ensure sanitized input for OpenCV
    assert isinstance(pts2d_bvs, np.ndarray)
    assert isinstance(pts3d, np.ndarray)
    pts2d_bvs = pts2d_bvs.astype(np.float64)
    pts3d = pts3d.astype(np.float64)

    # ransac p3p
    success, rvec, tvec, inliers = solvePnPRansac(
        pts3d,
        pts2d_bvs,
        cameraMatrix=np.eye(3),
        distCoeffs=None,
        iterationsCount=iterations_count,
        reprojectionError=ransac_thres,
        confidence=confidence,
        flags=SOLVEPNP_AP3P,
    )

    # there are situations where tvec is nan but the solver reports success
    if not success or np.any(np.isnan(tvec)):
        return None

    # refinement with just inliers
    inliers = inliers.ravel()
    rvec, tvec = solvePnPRefineLM(
        pts3d[inliers],
        pts2d_bvs[inliers],
        cameraMatrix=np.eye(3),
        distCoeffs=None,
        rvec=rvec,
        tvec=tvec,
    )
    R = Rodrigues(rvec)[0]
    t = tvec.ravel()
    return R, t, inliers


def camera_params_to_intrinsics_mat(camera_info):
    if camera_info["model"] != "SIMPLE_RADIAL":
        raise NotImplementedError

    K = np.array(
        [
            [camera_info["params"][0], 0, camera_info["params"][1]],
            [0, camera_info["params"][0], camera_info["params"][2]],
            [0, 0, 1],
        ]
    )
    return K
