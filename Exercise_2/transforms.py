import numpy as np


class Trafo:
    """
    Implements an affine transformation matrix

    Internally stores:
      rot_mat: 3x3 rotation matrix (accumulated)
      t_vec: 3-element translation vector (accumulated)

    The full transform is:  p' = rot_mat @ p + t_vec
    """

    def __init__(self):
        self.rot_mat = np.eye(3)   # identity rotation
        self.t_vec = np.zeros(3)   # zero translation


    def translate(self, t_vec: np.ndarray) -> None:
        """
        Adds the specified translation vector (t_vec) to the object's
        translation vector

        Args:
            t_vec: (3,) translation vector to add
        """
        self.t_vec = self.t_vec + t_vec


    def rotate(self, axis: np.ndarray, angle: float,
               center: np.ndarray) -> None:
        """
        Calculates a rotation matrix from the specified axis angle components
        and composes it the the object's rotation matrix

        Args:
            axis: (3,) rotation axis (need not be unit length)
            angle: rotation angle in radians
            center: (3,) point the axis passes through
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
        R = c * np.eye(3) + (1 - c) * np.outer(n, n) + s * K

        # Compose: new_rot = R @ old_rot
        self.rot_mat = R @ self.rot_mat

        # Compose: new_t = R @ old_t + center - R @ center
        self.t_vec = R @ self.t_vec + center - R @ center


    def xform_pnts(self, pts: np.ndarray) -> np.ndarray:
        """
        Transforms the incoming points w.r.t the object's 
        affine transformation

        Args:
            pts: (3, N) array of N 3-D points

        Returns:
            (3, N) array of transformed points
        """
        return self.rot_mat @ pts + self.t_vec[:, np.newaxis]


    def sys2sys(self, sys_src: np.ndarray, pts: np.ndarray) -> np.ndarray:
        """
        Converts the incoming points defined in sys_src coordinate frame
        to the coordinate frame specified by the object's rotation matrix
        and an origin defined by t_vec.

        Args:
            sys_src: (3, 3) rotation matrix of the source frame 
                    (columns = x, y, z axes in WCS).
            pts: (3, N) points expressed in sys_src frame

        Returns:
            (3, N) points expressed in the target (object's) frame
        """
        # Convert from source frame to WCS
        p_world = sys_src @ pts

        # Convert from WCS to target frame
        p_target = self.rot_mat.T @ (p_world - self.t_vec[:, np.newaxis])

        return p_target

