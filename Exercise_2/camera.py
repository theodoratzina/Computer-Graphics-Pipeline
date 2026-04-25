import numpy as np
from typing import Tuple


def perspective_project(
    pts: np.ndarray,
    focal: float,
    R: np.ndarray,
    t: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3-D world points onto the camera image plane using a
    pinhole perspective model.

    Pipeline:
        1. Transform pts from WCS to Camera Coordinate System (CCS):
               p_cam = R @ p_w + t
        2. Perspective divide (divide by depth = z_cam):
               x_2d = focal * x_cam / z_cam
               y_2d = focal * y_cam / z_cam

    Camera convention (same as in lookat):
        +X → right,  +Y → up,  -Z → viewing direction.
    Positive depth means the point is in FRONT of the camera (z_cam < 0).
    We use |z_cam| as the depth so that depth > 0 for visible points.

    Args:
        pts   : (3, N)  3-D points in WCS (non-homogeneous).
        focal : scalar  focal length in the same units as the CCS.
        R     : (3, 3)  rotation matrix  WCS → CCS  (from lookat).
        t     : (3,)    translation vector WCS → CCS (from lookat).

    Returns:
        pts_2d : (2, N)  projected 2-D coords on the camera plane.
                         Row 0 = x (horizontal), Row 1 = y (vertical).
        depths : (N,)    positive depth of each point  (= -z_cam).
    """
    # Step 1: world → camera coordinates
    # p_cam shape: (3, N)
    p_cam = R @ pts + t[:, np.newaxis]

    x_cam = p_cam[0]   # (N,)
    y_cam = p_cam[1]   # (N,)
    z_cam = p_cam[2]   # (N,)  negative for points in front of camera

    # Step 2: perspective divide
    # Camera looks along -Z, so depth = -z_cam (positive in front)
    depths = -z_cam                          # (N,)

    # Avoid division by zero (points at or behind camera plane)
    # Use z_cam directly; for correct projection z_cam should be < 0
    # depths = -z_cam > 0 for points in front of the camera,
    # so we divide by depths (equivalent to dividing by -z_cam)
    x_2d = focal * x_cam / depths
    y_2d = focal * y_cam / depths

    pts_2d = np.stack([x_2d, y_2d], axis=0) # (2, N)

    return pts_2d, depths


def rasterize(
    pts_2d: np.ndarray,
    plane_w: int,
    plane_h: int,
    res_w: int,
    res_h: int,
) -> np.ndarray:
    """
    Maps 2-D camera-plane coordinates to integer pixel coordinates.

    The camera plane is a rectangle of size plane_h x plane_w centred
    at the optical axis (origin). The image grid has res_h x res_w
    pixels, numbered:
        x (horizontal) : 0  (left)  →  res_w  (right)
        y (vertical)   : 0  (bottom) →  res_h  (top)
    i.e. the origin is at the BOTTOM-LEFT corner.

    Mapping (linear, centre-preserving):
        px = round( (x_plane + plane_w/2) / plane_w * res_w )
        py = round( (y_plane + plane_h/2) / plane_h * res_h )

    Args:
        pts_2d  : (2, N)  2-D coordinates on the camera plane.
                          Row 0 = x, Row 1 = y.
        plane_w : width  of the camera plane (world units).
        plane_h : height of the camera plane (world units).
        res_w   : horizontal resolution of the output image (pixels).
        res_h   : vertical   resolution of the output image (pixels).

    Returns:
        pix : (2, N) integer array.
              Row 0 = column index (x, 0=left).
              Row 1 = row    index (y, 0=bottom).
    """
    x_plane = pts_2d[0]   # (N,)
    y_plane = pts_2d[1]   # (N,)

    # Shift from [-plane_w/2, plane_w/2] to [0, plane_w], then scale
    px = np.round((x_plane + plane_w / 2.0) / plane_w * res_w).astype(int)
    py = np.round((y_plane + plane_h / 2.0) / plane_h * res_h).astype(int)

    return np.stack([px, py], axis=0)   # (2, N)
