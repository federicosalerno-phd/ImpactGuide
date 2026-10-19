"""Marker pose from one image: ChArUco corners and PnP.

The marker is a 5 x 5 ChArUco board (9 mm squares, 6 mm 4x4 markers, DICT_4X4_50) printed on a 50 mm aluminium plate,
as P["charuco"] in 04_code/tools/build_test_bench.py. Board frame as cv2.aruco.CharucoBoard: origin at the outer corner
of the first square (top left of the printed image), x along the printed columns, y along the rows, z into the board.
Poses are 4x4 matrices T_a_b (frame a from frame b) in mm.
"""
import cv2
import numpy as np

BOARD = {"squares": 5, "square_mm": 9.0, "marker_mm": 6.0, "dictionary": "DICT_4X4_50"}


def make_board(spec=BOARD):
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, spec["dictionary"]))
    n = spec["squares"]
    return cv2.aruco.CharucoBoard((n, n), spec["square_mm"], spec["marker_mm"], d)


def make_detector(board, K=None, dist=None):
    cp = cv2.aruco.CharucoParameters()
    if K is not None:   # interpolate the chessboard corners with the camera model instead of local homographies
        cp.cameraMatrix = np.asarray(K, np.float64)
        cp.distCoeffs = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
    cp.minMarkers = 1
    cp.tryRefineMarkers = True
    dp = cv2.aruco.DetectorParameters()
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.CharucoDetector(board, cp, dp)


def detect(detector, image):
    """ChArUco corners (N x 2, px) and their ids (N), and the number of ArUco markers found."""
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    corners, ids, _, marker_ids = detector.detectBoard(gray)
    n_markers = 0 if marker_ids is None else len(marker_ids)
    if ids is None:
        return np.empty((0, 2)), np.empty(0, np.int32), n_markers
    return corners.reshape(-1, 2).astype(np.float64), ids.ravel().astype(np.int32), n_markers


def matrix(rvec, tvec):
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(np.asarray(rvec, np.float64))[0]
    T[:3, 3] = np.ravel(tvec)
    return T


def estimate_pose(board, corners, ids, K, dist=None, min_corners=6):
    """Camera from board by PnP on the ChArUco corners: IPPE (planar), then Levenberg-Marquardt.

    Returns None with fewer than min_corners or when the solver finds no pose (degenerate corners), else a dict with T
    (4 x 4), rms_px, n (corners used) and ambiguity (ratio of the reprojection errors of the two IPPE solutions; near 1
    the mirrored pose is almost as good).
    """
    if len(ids) < max(min_corners, 4):
        return None
    K = np.asarray(K, np.float64)
    dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
    obj = board.getChessboardCorners()[ids].astype(np.float64)
    img = np.asarray(corners, np.float64)
    _, rvecs, tvecs, errs = cv2.solvePnPGeneric(obj, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
    errs = np.ravel(errs) if errs is not None else np.empty(0)
    if not len(rvecs) or len(errs) != len(rvecs):
        return None
    best = int(np.argmin(errs))
    rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvecs[best].copy(), tvecs[best].copy())
    proj = cv2.projectPoints(obj, rvec, tvec, K, dist)[0].reshape(-1, 2)
    rms = float(np.sqrt(np.mean(np.sum((proj - img) ** 2, axis=1))))
    if not np.isfinite(rms) or not np.all(np.isfinite(rvec)) or not np.all(np.isfinite(tvec)):
        return None
    ambiguity = float(errs[1 - best] / max(errs[best], 1e-12)) if len(errs) > 1 else float("inf")
    return {"T": matrix(rvec, tvec), "rms_px": rms, "n": len(ids), "ambiguity": ambiguity}


def reprojection_rms(board, corners, ids, T, K, dist=None):
    """RMS distance (px) between the corners and the board corners projected with the pose T."""
    obj = board.getChessboardCorners()[ids].astype(np.float64)
    proj = project(obj, T, K, dist)
    return float(np.sqrt(np.mean(np.sum((proj - corners) ** 2, axis=1))))


def transform(T, X):
    return np.asarray(X) @ T[:3, :3].T + T[:3, 3]


def project(X, T, K, dist=None):
    rvec = cv2.Rodrigues(T[:3, :3])[0]
    dist = np.zeros(5) if dist is None else np.asarray(dist, np.float64)
    return cv2.projectPoints(np.asarray(X, np.float64), rvec, T[:3, 3], np.asarray(K, np.float64), dist)[0].reshape(-1, 2)


def rotation_angle_deg(R):
    return float(np.degrees(np.linalg.norm(cv2.Rodrigues(R)[0])))   # accurate for small angles, unlike arccos


def pose_error(T_est, T_gt):
    """Rotation error (deg) and translation error (mm, norm and along the optical axis) of camera from board."""
    dt = T_est[:3, 3] - T_gt[:3, 3]
    return {"rot_deg": rotation_angle_deg(T_est[:3, :3] @ T_gt[:3, :3].T), "trans_mm": float(np.linalg.norm(dt)),
            "depth_mm": float(dt[2])}


def target_error(T_est, T_gt, X, K, dist=None):
    """Error that the pose error puts on target points X (board frame): 3D (mm) and in the image (px), RMS and max."""
    d3 = np.linalg.norm(transform(T_est, X) - transform(T_gt, X), axis=1)
    d2 = np.linalg.norm(project(X, T_est, K, dist) - project(X, T_gt, K, dist), axis=1)
    return {"tre_rms_mm": float(np.sqrt(np.mean(d3 ** 2))), "tre_max_mm": float(d3.max()),
            "overlay_rms_px": float(np.sqrt(np.mean(d2 ** 2))), "overlay_max_px": float(d2.max())}
