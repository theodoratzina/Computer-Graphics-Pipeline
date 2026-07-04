import numpy as np
from utils import vector_interp
from lighting import light


def calc_normals(pts: np.ndarray, t_pos_idx: np.ndarray) -> np.ndarray:
    """
    Compute a unit normal vector per vertex of a triangle mesh.

    Args:
        pts       : (3, Nv) coordinates of the mesh vertices (WCS)
        t_pos_idx : (NT, 3) triangle vertex indices (0-based)

    Returns:
        nrm : (3, Nv) array with the unit normal of every vertex
    """
    pts = np.asarray(pts, dtype=float)
    if pts.shape[0] != 3 and pts.shape[1] == 3:
        pts = pts.T

    faces = np.asarray(t_pos_idx)
    if faces.ndim == 2 and faces.shape[1] != 3 and faces.shape[0] == 3:
        faces = faces.T
    faces = faces.astype(int)

    Nv = pts.shape[1]
    P = pts.T   # (Nv, 3)

    v0 = P[faces[:, 0]]
    v1 = P[faces[:, 1]]
    v2 = P[faces[:, 2]]

    face_n = np.cross(v1 - v0, v2 - v0)    # (NT, 3), area-weighted

    normals = np.zeros((Nv, 3), dtype=float)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_n)

    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths[lengths < 1e-12] = 1.0
    normals = normals / lengths

    return normals.T


def _sample_texture(uv: np.ndarray, tex: np.ndarray) -> np.ndarray:
    """
    Nearest-neighbour lookup of a colour from a texture image.
 
    Args:
        uv  : (2,) normalized texture coordinates (u, v) in [0, 1]
        tex : (K, L, 3) texture image, float in [0, 1]
 
    Returns:
        (3,) RGB colour sampled at uv
    """
    K, L = tex.shape[0], tex.shape[1]
    col = int(np.round(uv[0] * L)) % L
    row = int(np.round(uv[1] * K)) % K
    return tex[row, col]


def _scanline_bounds(vy, img_h):
    """
    Clamp a triangle's vertical extent to the visible rows of the canvas.
 
    Args:
        vy    : (3,) y (row) pixel coordinates of the triangle vertices
        img_h : int, canvas height in pixels
 
    Returns:
        (ymin, ymax) : int scanline range, clipped to [0, img_h - 1]
    """
    ymin = max(0, int(np.min(vy)))
    ymax = min(img_h - 1, int(np.max(vy)))
    return ymin, ymax


def shade_gouraud(v_pos: np.ndarray, v_nrm: np.ndarray, v_uvs: np.ndarray,
                  tex: np.ndarray, cam_pos: np.ndarray, ka: float, kd: float,
                  ks: float, n: float, l_pos, l_int, l_amb: np.ndarray,
                  img: np.ndarray, bcoords: np.ndarray) -> np.ndarray:
    """
    Shade a single triangle with Gouraud shading (vectorized scanline).
 
    The full illumination model is evaluated only at the 3 vertices, the
    resulting vertex colours are then linearly interpolated across the
    triangle (first along the active edges, then horizontally per scanline).
 
    Args:
        v_pos   : (3, 2) projected pixel coords [x, y] of the 3 vertices
        v_nrm   : (3, 3) per-vertex unit normals (row i -> vertex i)
        v_uvs   : (3, 2) per-vertex texture coordinates
        tex     : (K, L, 3) texture image, float in [0, 1]
        cam_pos : (3,) camera position in WCS
        ka, kd, ks, n : Phong material coefficients
        l_pos, l_int  : point light positions / intensities (N, 3)
        l_amb   : (3,) ambient light intensity
        img     : (res_h, res_w, 3) canvas with any pre-existing triangles
        bcoords : (3,) triangle barycenter in 3D (pre-projection), used to fix
                  the L and V directions of the lighting model
 
    Returns:
        updated_img : (res_h, res_w, 3) canvas with this triangle painted on
                      top of the pre-existing content (written in place)
    """
    updated_img = img   # write in place (painter's order), no per-triangle copy

    # Colour at each vertex from the full illumination model
    vcolors = np.zeros((3, 3), dtype=float)
    for i in range(3):
        albedo = _sample_texture(v_uvs[i], tex)
        vcolors[i] = light(bcoords, v_nrm[i], albedo, cam_pos,
                           ka, kd, ks, n, l_pos, l_int, l_amb)

    vx = v_pos[:, 0]
    vy = v_pos[:, 1]
    ymin, ymax = _scanline_bounds(vy, updated_img.shape[0])
    edges = [(0, 1), (1, 2), (2, 0)]

    for y in range(ymin, ymax + 1):
        x_intersections = []
        c_intersections = []
        for (i, j) in edges:
            y1, y2 = vy[i], vy[j]
            x1, x2 = vx[i], vx[j]
            if y1 == y2:
                continue
            ek_min, ek_max = min(y1, y2), max(y1, y2)
            if y < ek_min or y > ek_max:
                continue
            if y == ek_max and ek_max != ymax:
                continue
            t = (y - y1) / (y2 - y1)
            t = np.clip(t, 0.0, 1.0)
            x_intersections.append(x1 + t * (x2 - x1))
            p_i = np.array([x1, y1]); p_j = np.array([x2, y2])
            c_intersections.append(
                vector_interp(p_i, p_j, vcolors[i], vcolors[j], y, dim=2))

        if len(x_intersections) < 2:
            continue

        pairs = sorted(zip(x_intersections, c_intersections), key=lambda p: p[0])
        x_A, c_A = pairs[0]
        x_B, c_B = pairs[-1]

        x_start = max(0, int(np.ceil(x_A)))
        x_end = min(updated_img.shape[1] - 1, int(x_B))
        if x_start > x_end:
            continue

        # Vectorized horizontal fill from A to B
        xs = np.arange(x_start, x_end + 1)
        if np.isclose(x_B - x_A, 0.0):
            tv = np.full(xs.shape, 0.5)
        else:
            tv = np.clip((xs - x_A) / (x_B - x_A), 0.0, 1.0)
        span = (1.0 - tv)[:, None] * c_A + tv[:, None] * c_B
        updated_img[y, x_start:x_end + 1] = span

    return updated_img


def shade_phong(v_pos: np.ndarray, v_nrm: np.ndarray, v_uvs: np.ndarray,
                tex: np.ndarray, cam_pos: np.ndarray, ka: float, kd: float,
                ks: float, n: float, l_pos, l_int, l_amb: np.ndarray,
                img: np.ndarray, bcoords: np.ndarray) -> np.ndarray:
    """
    Shade a single triangle with Phong shading (vectorized scanline).
 
    The per-vertex normals and texture coordinates are linearly interpolated
    across the triangle (along the active edges, then horizontally), the full
    illumination model is then evaluated per pixel from the interpolated
    normal and the texture-sampled albedo. Each scanline span is shaded with a
    single batched call to light().
 
    Args:
        v_pos   : (3, 2) projected pixel coords [x, y] of the 3 vertices
        v_nrm   : (3, 3) per-vertex unit normals (row i -> vertex i)
        v_uvs   : (3, 2) per-vertex texture coordinates
        tex     : (K, L, 3) texture image, float in [0, 1]
        cam_pos : (3,) camera position in WCS
        ka, kd, ks, n : Phong material coefficients
        l_pos, l_int  : point light positions / intensities (N, 3)
        l_amb   : (3,) ambient light intensity
        img     : (res_h, res_w, 3) canvas with any pre-existing triangles
        bcoords : (3,) triangle barycenter in 3D (pre-projection), used to fix
                  the L and V directions of the lighting model
 
    Returns:
        updated_img : (res_h, res_w, 3) canvas with this triangle painted on
                      top of the pre-existing content (written in place)
    """
    updated_img = img   # write in place (painter's order), no per-triangle copy

    vx = v_pos[:, 0]
    vy = v_pos[:, 1]
    ymin, ymax = _scanline_bounds(vy, updated_img.shape[0])
    edges = [(0, 1), (1, 2), (2, 0)]

    K, L = tex.shape[0], tex.shape[1]

    for y in range(ymin, ymax + 1):
        x_intersections = []
        n_intersections = []
        uv_intersections = []
        for (i, j) in edges:
            y1, y2 = vy[i], vy[j]
            x1, x2 = vx[i], vx[j]
            if y1 == y2:
                continue
            ek_min, ek_max = min(y1, y2), max(y1, y2)
            if y < ek_min or y > ek_max:
                continue
            if y == ek_max and ek_max != ymax:
                continue
            t = (y - y1) / (y2 - y1)
            t = np.clip(t, 0.0, 1.0)
            x_intersections.append(x1 + t * (x2 - x1))
            p_i = np.array([x1, y1]); p_j = np.array([x2, y2])
            n_intersections.append(
                vector_interp(p_i, p_j, v_nrm[i], v_nrm[j], y, dim=2))
            uv_intersections.append(
                vector_interp(p_i, p_j, v_uvs[i], v_uvs[j], y, dim=2))

        if len(x_intersections) < 2:
            continue

        pairs = sorted(zip(x_intersections, n_intersections, uv_intersections),
                       key=lambda p: p[0])
        x_A, n_A, uv_A = pairs[0]
        x_B, n_B, uv_B = pairs[-1]

        x_start = max(0, int(np.ceil(x_A)))
        x_end = min(updated_img.shape[1] - 1, int(x_B))
        if x_start > x_end:
            continue

        # Vectorized: interpolate normal + uv across the whole span
        xs = np.arange(x_start, x_end + 1)
        if np.isclose(x_B - x_A, 0.0):
            tv = np.full(xs.shape, 0.5)
        else:
            tv = np.clip((xs - x_A) / (x_B - x_A), 0.0, 1.0)

        nrm_span = (1.0 - tv)[:, None] * n_A + tv[:, None] * n_B    # (M, 3)
        uv_span = (1.0 - tv)[:, None] * uv_A + tv[:, None] * uv_B   # (M, 2)

        cols = np.round(uv_span[:, 0] * L).astype(int) % L
        rows = np.round(uv_span[:, 1] * K).astype(int) % K
        albedo_span = tex[rows, cols]   # (M, 3)

        # One batched lighting call for the span, light() re-normalises the
        # interpolated normals and keeps L/V fixed by bcoords
        updated_img[y, x_start:x_end + 1] = light(
            bcoords, nrm_span, albedo_span, cam_pos,
            ka, kd, ks, n, l_pos, l_int, l_amb)

    return updated_img
