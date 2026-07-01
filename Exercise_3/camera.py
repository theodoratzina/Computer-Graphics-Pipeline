import numpy as np
from typing import Tuple


def lookat(eye: np.ndarray, up: np.ndarray,
           target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the view-matrix parameters (R, t) that transform points
    from the World Coordinate System (WCS) into the Camera Coordinate
    System (CCS).

    Camera convention:
      +X  →  right
      +Y  →  up
      -Z  →  viewing direction  (camera looks along -Z in CCS)

    Args:
        eye: (3,) camera centre in WCS
        up: (3,) approximate up direction (need not be unit)
        target: (3,) point the camera looks at

    Returns:
        R: (3, 3) rotation matrix (WCS → CCS)
        t: (3,) translation vector (WCS → CCS)
    """
    # -Z axis of camera (points from target toward eye)
    z_cam = eye - target
    z_cam = z_cam / np.linalg.norm(z_cam)

    # +X axis: perpendicular to both up and z_cam
    x_cam = np.cross(up, z_cam)
    x_cam = x_cam / np.linalg.norm(x_cam)

    # +Y axis: recompute to guarantee orthogonality
    y_cam = np.cross(z_cam, x_cam)
    # y_cam is already unit since z_cam and x_cam are orthonormal

    # Build rotation matrix
    R = np.stack([x_cam, y_cam, z_cam], axis=0)

    # Translation: brings the camera origin to the WCS origin
    t = -R @ eye

    return R, t


def perspective_project(pts: np.ndarray, focal: float, R: np.ndarray,
                        t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Projects 3-D world points onto the camera image plane using a
    pinhole perspective model

    Camera convention (same as in lookat):
        +X → right,  +Y → up,  -Z → viewing direction
    Positive depth means the point is in front of the camera

    Args:
        pts: (3, N) 3-D points in WCS (non-homogeneous)
        focal: scalar focal length in the same units as the CCS
        R: (3, 3) rotation matrix  WCS → CCS  (from lookat)
        t: (3,) translation vector WCS → CCS (from lookat)

    Returns:
        pts_2d: (2, N) projected 2-D coords on the camera plane
        depth: (N,) positive depth of each point (= -z_cam)
    """
    # World → Camera coordinates
    p_cam = R @ pts + t[:, np.newaxis]

    x_cam = p_cam[0]
    y_cam = p_cam[1]
    z_cam = p_cam[2]

    # Perspective divide
    depth = -z_cam

    # Avoid division by zero (points at or behind camera plane)
    x_2d = focal * x_cam / depth
    y_2d = focal * y_cam / depth

    pts_2d = np.stack([x_2d, y_2d], axis=0)

    return pts_2d, depth


def rasterize(pts_2d: np.ndarray, plane_w: int, plane_h: int,
              res_w: int, res_h: int) -> np.ndarray:
    """
    Maps 2-D camera-plane coordinates to integer pixel coordinates

    Args:
        pts_2d: (2, N) 2-D coordinates on the camera plane
        plane_w: width of the camera plane (world units)
        plane_h: height of the camera plane (world units)
        res_w: horizontal resolution of the output image (pixels)
        res_h: vertical resolution of the output image (pixels)

    Returns:
        pix: (2, N) integer array
    """
    x_plane = pts_2d[0]
    y_plane = pts_2d[1]

    # Shift from [-plane_w/2, plane_w/2] to [0, plane_w], then scale
    px = np.round((x_plane + plane_w / 2.0) / plane_w * res_w).astype(int)
    py = np.round((y_plane + plane_h / 2.0) / plane_h * res_h).astype(int)

    return np.stack([px, py], axis=0)