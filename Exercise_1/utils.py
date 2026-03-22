import numpy as np


def vector_interp(p1, p2, V1, V2, coord, dim):
    """
    Linear interpolation of vectors V1, V2 defined at points p1 and p2.
    
    Args:
        p1    : (2,) array, coordinates of first point
        p2    : (2,) array, coordinates of second point
        V1    : (d,) array, vector value at p1
        V2    : (d,) array, vector value at p2
        coord : float, the known coordinate (x or y) of the target point p
        dim   : int, 1 if coord is x (index 0), 2 if coord is y (index 1)
    
    Returns:
        V : (d,) array, interpolated vector at p
    """
    # Ensure inputs are numpy arrays
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    V1 = np.asarray(V1, dtype=float)
    V2 = np.asarray(V2, dtype=float)

    # Convert to zero-based index (0 for x, 1 for y)
    idx = dim - 1

    # Get the coordinates of p1 and p2 in the specified dimension
    c1 = p1[idx]
    c2 = p2[idx]

    # If both points share the same coordinate, no interpolation needed
    if np.isclose(c2 - c1, 0.0):
        t = 0.5  # Take the average
    else:
        t = (coord - c1) / (c2 - c1)  # Compute interpolation factor
        t = np.clip(t, 0.0, 1.0)      # Clamp t to [0, 1] to avoid extrapolation

    # Perform linear interpolation
    V = (1 - t) * V1 + t * V2

    return V
