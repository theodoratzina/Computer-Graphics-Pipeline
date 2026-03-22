import numpy as np
from utils import vector_interp


def f_shading(img, vertices, vcolors):
    """
    Fill a triangle with flat shading (uniform mean vertex color). The scanline
    filling algorithm is used, with the boundary convention that each pixel
    belongs to the triangle that is higher and/or more to the right.

    Args:
        img      : (M, N, 3) float array, canvas with any pre-existing triangles
        vertices : (3, 2) int array, [x, y] coordinates of each vertex
        vcolors  : (3, 3) float array, RGB colors per vertex in [0, 1]

    Returns:
        updated_img : (M, N, 3) float array, canvas with the new triangle painted
    """
    # Create a copy of the input image to modify
    updated_img = img.copy()

    # Compute the mean color of the triangle vertices
    mean_color = np.mean(vcolors, axis=0)

    # Separate vertex x and y coordinates
    vx = vertices[:, 0]
    vy = vertices[:, 1]

    # Scanline filling algorithm from lowest to highest vertex
    ymin = max(0, int(np.min(vy)))
    ymax = min(img.shape[0] - 1, int(np.max(vy)))

    # Triangle edges as pairs of vertex indices
    edges = [(0, 1), (1, 2), (2, 0)]

    for y in range(ymin, ymax + 1):
        # Find intersections of the scanline with triangle edges
        x_intersections = []

        for (i, j) in edges:
            y1, y2 = vy[i], vy[j]
            x1, x2 = vx[i], vx[j]

            if y1 == y2:
                continue  # Skip horizontal edges

            ek_min = min(y1, y2)
            ek_max = max(y1, y2)
 
            # Active scanline range: [ek_min, ek_max)
            if y < ek_min or y > ek_max:
                continue
            if y == ek_max and ek_max != ymax:
                continue   # Skip the top endpoint for all but the global topmost scanline
 
            # Compute the x-coordinate where this edge crosses the scanline y
            t = (y - y1) / (y2 - y1)
            t = np.clip(t, 0.0, 1.0)   # Ensure t is within [0, 1]
            x_int = x1 + t * (x2 - x1)
            x_intersections.append(x_int)

        # Need exactly 2 crossing points to define a fill interval
        if len(x_intersections) < 2:
            continue
 
        # Sort boundaries left to right
        x_intersections.sort()
        x_start = max(0, int(np.ceil(x_intersections[0])))
        x_end   = min(updated_img.shape[1] - 1, int(x_intersections[-1]))
 
        # Paint the entire span with the mean color
        if x_start <= x_end:
            updated_img[y, x_start:x_end + 1] = mean_color

    return updated_img


def g_shading(img, vertices, vcolors):
    """
    Fill a triangle with Gouraud shading (bilinear color interpolation).
    Colors are interpolated along the active edges to find the colors at the
    left (A) and right (B) boundary points per scanline, then interpolated
    horizontally from A to B for each pixel P = (x, y).
 
    Args:
        img      : (M, N, 3) float array, canvas with any pre-existing triangles
        vertices : (3, 2) int array, [x, y] coordinates of each vertex
        vcolors  : (3, 3) float array, RGB colors per vertex in [0, 1]
 
    Returns:
        updated_img : (M, N, 3) float array, canvas with the new triangle painted
    """
    # Create a copy of the input image to modify
    updated_img = img.copy()
 
    # Separate vertex x and y coordinates
    vx = vertices[:, 0]
    vy = vertices[:, 1]
 
    # Scanline filling algorithm from lowest to highest vertex
    ymin = max(0, int(np.min(vy)))
    ymax = min(img.shape[0] - 1, int(np.max(vy)))
 
    # Triangle edges as pairs of vertex indices
    edges = [(0, 1), (1, 2), (2, 0)]
 
    for y in range(ymin, ymax + 1):
        # Each active edge contributes one boundary point and its interpolated color
        x_intersections = []
        c_intersections = []
 
        for (i, j) in edges:
            y1, y2 = vy[i], vy[j]
            x1, x2 = vx[i], vx[j]
 
            if y1 == y2:
                continue  # Skip horizontal edges
 
            ek_min = min(y1, y2)
            ek_max = max(y1, y2)
 
            # Active scanline range: [ek_min, ek_max)
            if y < ek_min or y > ek_max:
                continue
            if y == ek_max and ek_max != ymax:
                continue  # Skip the top endpoint for all but the global topmost scanline
 
            # Compute the x-coordinate where this edge crosses the scanline y
            t = (y - y1) / (y2 - y1)
            t = np.clip(t, 0.0, 1.0)  # Ensure t is within [0, 1]
            x_int = x1 + t * (x2 - x1)
            x_intersections.append(x_int)
 
            # First Pass: Interpolate color along the edge at scanline y
            p_i = np.array([x1, y1])
            p_j = np.array([x2, y2])
            c_int = vector_interp(p_i, p_j, vcolors[i], vcolors[j], y, dim=2)
            c_intersections.append(c_int)
 
        # Need exactly 2 crossing points to define a fill interval
        if len(x_intersections) < 2:
            continue
 
        # Sort boundaries left to right, keeping colors paired with their x
        paires = sorted(zip(x_intersections, c_intersections), key=lambda p: p[0])
        x_A, c_A = paires[0]
        x_B, c_B = paires[-1]

        # A is the left boundary point, B is the right boundary point
        p_A = np.array([x_A, y])
        p_B = np.array([x_B, y])
 
        # Clamp to canvas bounds
        x_start = max(0, int(np.ceil(x_A)))   # Ceiling to start filling from the first pixel inside the boundary
        x_end   = min(updated_img.shape[1] - 1, int(x_B))

        if x_start > x_end:
            continue
 
        # Second pass: Interpolate color horizontally from A to B for each pixel
        for x in range(x_start, x_end + 1):
            color = vector_interp(p_A, p_B, c_A, c_B, x, dim=1)
            updated_img[y, x] = color

        # # Second pass: vectorized horizontal interpolation from A to B
        # xs = np.arange(x_start, x_end + 1, dtype=float)
        # if np.isclose(x_B - x_A, 0.0):
        #     t_vals = np.zeros(len(xs))
        # else:
        #     t_vals = (xs - x_A) / (x_B - x_A)
        #     t_vals = np.clip(t_vals, 0.0, 1.0)  # Ensure t is within [0, 1]

        # color = (1.0 - t_vals[:, None]) * c_A + t_vals[:, None] * c_B
        # updated_img[y, x_start:x_end + 1] = color
 
    return updated_img


def t_shading(img, vertices, uv, textImg):
    """
    Fill a triangle with texture shading (UV-mapped texture interpolation).
    UV coordinates are interpolated along the active edges to find UV at the
    left (A) and right (B) boundary points per scanline, then interpolated
    horizontally from A to B for each pixel P = (x, y). The resulting UV
    coordinates are used to look up the color from textImg.
 
    Args:
        img      : (M, N, 3) float array, canvas with any pre-existing triangles
        vertices : (3, 2) int array, [x, y] coordinates of each vertex
        uv       : (3, 2) float array, texture coordinates (u, v) normalized in [0, 1]
        textImg  : (K, L, 3) float array, texture image to sample from
 
    Returns:
        updated_img : (M, N, 3) float array, canvas with the new triangle painted
    """
    # Create a copy of the input image to modify
    updated_img = img.copy()
 
    # Texture image dimensions
    K, L = textImg.shape[0], textImg.shape[1]
 
    # Separate vertex x and y coordinates
    vx = vertices[:, 0]
    vy = vertices[:, 1]
 
    # Scanline filling algorithm from lowest to highest vertex
    ymin = max(0, int(np.min(vy)))
    ymax = min(img.shape[0] - 1, int(np.max(vy)))
 
    # Triangle edges as pairs of vertex indices
    edges = [(0, 1), (1, 2), (2, 0)]
 
    for y in range(ymin, ymax + 1):
        # Each active edge contributes one boundary point and its interpolated UV
        x_intersections  = []
        uv_intersections = []
 
        for (i, j) in edges:
            y1, y2 = vy[i], vy[j]
            x1, x2 = vx[i], vx[j]
 
            if y1 == y2:
                continue  # Skip horizontal edges
 
            ek_min = min(y1, y2)
            ek_max = max(y1, y2)
 
            # Active scanline range: [ek_min, ek_max)
            if y < ek_min or y > ek_max:
                continue
            if y == ek_max and ek_max != ymax:
                continue  # Skip the top endpoint for all but the global topmost scanline
 
            # Compute the x-coordinate where this edge crosses the scanline y
            t = (y - y1) / (y2 - y1)
            t = np.clip(t, 0.0, 1.0)  # Ensure t is within [0, 1]
            x_int = x1 + t * (x2 - x1)
            x_intersections.append(x_int)
 
            # First pass: interpolate UV coordinates along the edge at scanline y
            p_i  = np.array([x1, y1])
            p_j  = np.array([x2, y2])
            uv_int = vector_interp(p_i, p_j, uv[i], uv[j], y, dim=2)
            uv_intersections.append(uv_int)
 
        # Need exactly 2 crossing points to define a fill interval
        if len(x_intersections) < 2:
            continue
 
        # Sort boundaries left to right, keeping UV paired with their x
        paires = sorted(zip(x_intersections, uv_intersections), key=lambda p: p[0])
        x_A, uv_A = paires[0]
        x_B, uv_B = paires[-1]

        # A is the left boundary point, B is the right boundary point
        p_A = np.array([x_A, y])
        p_B = np.array([x_B, y])
 
        # Clamp to canvas bounds
        x_start = max(0, int(np.ceil(x_A)))  # Ceiling to start filling from the first pixel inside the boundary
        x_end   = min(updated_img.shape[1] - 1, int(x_B))

        if x_start > x_end:
            continue

        # Second pass: interpolate UV horizontally from A to B for each pixel
        for x in range(x_start, x_end + 1):
            uv_p = vector_interp(p_A, p_B, uv_A, uv_B, x, dim=1)
 
            # Map normalized UV coordinates to texture image pixel indices
            tex_col = np.round(uv_p[0] * L).astype(int) % L
            tex_row = np.round(uv_p[1] * K).astype(int) % K
 
            # Look up and copy the color from the texture image
            updated_img[y, x] = textImg[tex_row, tex_col]

        # # Second pass: vectorized horizontal UV interpolation from A to B
        # xs = np.arange(x_start, x_end + 1, dtype=float)
        # if np.isclose(x_B - x_A, 0.0):
        #     t_vals = np.zeros(len(xs))
        # else:
        #     t_vals = (xs - x_A) / (x_B - x_A)
        #     t_vals = np.clip(t_vals, 0.0, 1.0)

        # uvs_row = (1.0 - t_vals[:, None]) * uv_A + t_vals[:, None] * uv_B  # (N, 2)

        # tex_cols = np.round(uvs_row[:, 0] * L).astype(int) % L
        # tex_rows = np.round(uvs_row[:, 1] * K).astype(int) % K

        # updated_img[y, x_start:x_end + 1] = textImg[tex_rows, tex_cols]
 
    return updated_img

