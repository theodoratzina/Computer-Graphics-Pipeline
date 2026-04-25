import numpy as np
from typing import Tuple


class Trafo:
    """
    Implements an affine transformation (rotation + translation).
    Internally stores:
      rot_mat : 3x3 rotation matrix (accumulated)
      t_vec   : 3-element translation vector (accumulated)
    The full transform is:  p' = rot_mat @ p + t_vec
    """

    def __init__(self):
        self.rot_mat = np.eye(3)       # identity rotation
        self.t_vec   = np.zeros(3)     # zero translation

    # ------------------------------------------------------------------
    def translate(self, t_vec: np.ndarray) -> None:
        """
        Adds t_vec to the object's accumulated translation vector.

        Args:
            t_vec : (3,) translation vector to add.
        """
        self.t_vec = self.t_vec + t_vec

    # ------------------------------------------------------------------
    def rotate(self, axis: np.ndarray, angle: float,
               center: np.ndarray) -> None:
        """
        Builds a rotation matrix via Rodrigues' formula for a
        RIGHT-HANDED (clockwise when viewed along +axis) rotation
        by `angle` radians around the direction given by `axis`,
        passing through `center`.  The new rotation is LEFT-composed
        with the existing rot_mat (i.e. applied AFTER).

        Math:
            R = cos(a)*I + (1-cos(a))*n*nT + sin(a)*[n]x
        where n is the unit axis vector and [n]x is the skew-symmetric
        cross-product matrix.

        For a rotation about a point `center`:
            p' = R @ (p - center) + center
               = R @ p  +  (center - R @ center)

        Composition with existing transform  p_1 = rot_mat @ p + t_vec :
            p_2 = R_new @ p_1 + (center - R_new @ center)
                = (R_new @ rot_mat) @ p
                  + R_new @ t_vec + center - R_new @ center

        Args:
            axis   : (3,) rotation axis (need not be unit length).
            angle  : rotation angle in radians.
            center : (3,) point the axis passes through.
        """
        # Normalise axis
        n = axis / np.linalg.norm(axis)

        c, s = np.cos(angle), np.sin(angle)

        # Skew-symmetric cross-product matrix of n
        K = np.array([
            [ 0,    -n[2],  n[1]],
            [ n[2],  0,    -n[0]],
            [-n[1],  n[0],  0   ]
        ])

        # Rodrigues' formula
        R_new = c * np.eye(3) + (1 - c) * np.outer(n, n) + s * K

        # Compose: new_rot = R_new @ old_rot
        self.rot_mat = R_new @ self.rot_mat

        # Compose: new_t = R_new @ old_t + center - R_new @ center
        self.t_vec = R_new @ self.t_vec + center - R_new @ center

    # ------------------------------------------------------------------
    def xform_pnts(self, pts: np.ndarray) -> np.ndarray:
        """
        Applies the affine transform  p' = rot_mat @ p + t_vec
        to each column of pts.

        Args:
            pts : (3, N) array of N 3-D points.

        Returns:
            (3, N) array of transformed points.
        """
        # pts shape: (3, N)
        return self.rot_mat @ pts + self.t_vec[:, np.newaxis]

    # ------------------------------------------------------------------
    def sys2sys(self, sys_src: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Converts pts from the coordinate frame described by `sys_src`
        to the coordinate frame described by (rot_mat, t_vec).

        `sys_src` columns are the basis vectors of the source frame
        expressed in the world frame (WCS). The origin of the source
        frame is assumed to be at the world origin.

        Step 1 – source frame → WCS:
            p_world = sys_src @ p_src

        Step 2 – WCS → target frame (rot_mat, t_vec):
            p_target = rot_mat.T @ (p_world - t_vec)

        Args:
            sys_src : (3, 3) rotation matrix of the source frame
                      (columns = x, y, z axes in WCS).
            pts     : (3, N) points expressed in sys_src frame.

        Returns:
            (3, N) points expressed in the target (object's) frame.
        """
        # Convert from source frame to WCS
        p_world = sys_src @ pts                    # (3, N)

        # Convert from WCS to target frame
        p_target = self.rot_mat.T @ (p_world - self.t_vec[:, np.newaxis])

        return p_target


# ======================================================================
def lookat(eye: np.ndarray, up: np.ndarray,
           target: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes the view-matrix parameters (R, t) that transform points
    from the World Coordinate System (WCS) into the Camera Coordinate
    System (CCS).

    The camera convention used here:
      +X  →  right
      +Y  →  up
      -Z  →  viewing direction  (camera looks along -Z in CCS)

    Derivation:
        z_cam = normalize(eye - target)   # points AWAY from target
        x_cam = normalize(up  × z_cam)   # right vector  (corrected up)
        y_cam = z_cam × x_cam            # true up vector

        R = [ x_cam ]   (rows are the camera axes expressed in WCS)
            [ y_cam ]
            [ z_cam ]

    A world point p_w maps to camera coords via:
        p_c = R @ (p_w - eye) = R @ p_w + t,   where t = -R @ eye

    Args:
        eye    : (3,) camera centre in WCS.
        up     : (3,) approximate up direction (need not be unit).
        target : (3,) point the camera looks at.

    Returns:
        R : (3, 3) rotation matrix  (WCS → CCS)
        t : (3,)   translation vector  (WCS → CCS),  t = -R @ eye
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

    # Build rotation matrix  (rows = camera basis vectors in WCS)
    R = np.stack([x_cam, y_cam, z_cam], axis=0)   # (3, 3)

    # Translation: brings the camera origin to the WCS origin
    t = -R @ eye                                    # (3,)

    return R, t
