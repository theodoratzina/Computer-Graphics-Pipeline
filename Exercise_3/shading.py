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
    # Accept (Nv, 3) too and bring it to the required (3, Nv) layout
    if pts.shape[0] != 3 and pts.shape[1] == 3:
        pts = pts.T

    faces = np.asarray(t_pos_idx)
    # Accept (3, NT) too and bring it to the (NT, 3) layout used in hw1/hw2
    if faces.ndim == 2 and faces.shape[1] != 3 and faces.shape[0] == 3:
        faces = faces.T
    faces = faces.astype(int)

    Nv = pts.shape[1]
    P = pts.T   # (Nv, 3) row form, convenient for gather

    # Gather the three vertices of every triangle
    v0 = P[faces[:, 0]]   # (NT, 3)
    v1 = P[faces[:, 1]]
    v2 = P[faces[:, 2]]

    # Face normal via right-hand rule; magnitude = 2 * triangle area, so summing
    # the raw (non-normalised) vectors gives an area-weighted vertex normal
    face_n = np.cross(v1 - v0, v2 - v0)    # (NT, 3)

    # Accumulate every face normal onto its three vertices
    normals = np.zeros((Nv, 3), dtype=float)
    for k in range(3):
        np.add.at(normals, faces[:, k], face_n)

    # Normalise each vertex normal to unit length (guard zero-length vertices)
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
    col = int(np.round(uv[0] * L)) % L   # u -> column
    row = int(np.round(uv[1] * K)) % K   # v -> row
    return tex[row, col]


def shade_gouraud(v_pos: np.ndarray, v_nrm: np.ndarray, v_uvs: np.ndarray,
                  tex: np.ndarray, cam_pos: np.ndarray, ka: float, kd: float,
                  ks: float, n: float, l_pos, l_int, l_amb: np.ndarray,
                  img: np.ndarray, bcoords: np.ndarray) -> np.ndarray:
    """
    Shade a single triangle with Gouraud shading.

    Args:
        v_pos   : (3, 2) projected pixel coords [x, y] of the 3 vertices
        v_nrm   : (3, 3) per-vertex unit normals (row i -> vertex i)
        v_uvs   : (3, 2) per-vertex texture coordinates
        tex     : (K, L, 3) texture image, float in [0, 1]
        cam_pos : (3,) camera position in WCS
        ka, kd, ks, n : Phong material coefficients
        l_pos, l_int : point light positions / intensities (N, 3)
        l_amb   : (3,) ambient light intensity
        img     : (res_h, res_w, 3) canvas with any pre-existing triangles
        bcoords : (3,) triangle barycenter in 3D (pre-projection), used to fix
                  the L and V directions of the lighting model

    Returns:
        updated_img : (res_h, res_w, 3) canvas with this triangle painted
    """
    updated_img = img.copy()

    # Colour at each vertex from the full illumination model
    vcolors = np.zeros((3, 3), dtype=float)
    for i in range(3):
        albedo = _sample_texture(v_uvs[i], tex)
        vcolors[i] = light(bcoords, v_nrm[i], albedo, cam_pos,
                           ka, kd, ks, n, l_pos, l_int, l_amb)

    # Scanline fill, interpolating the 3 vertex colours (hw1 g_shading)
    vx = v_pos[:, 0]
    vy = v_pos[:, 1]

    ymin = max(0, int(np.min(vy)))
    ymax = min(updated_img.shape[0] - 1, int(np.max(vy)))

    edges = [(0, 1), (1, 2), (2, 0)]

    for y in range(ymin, ymax + 1):
        x_intersections = []
        c_intersections = []

        for (i, j) in edges:
            y1, y2 = vy[i], vy[j]
            x1, x2 = vx[i], vx[j]

            if y1 == y2:
                continue   # skip horizontal edges

            ek_min, ek_max = min(y1, y2), max(y1, y2)
            if y < ek_min or y > ek_max:
                continue
            if y == ek_max and ek_max != ymax:
                continue   # boundary convention (higher/right)

            t = (y - y1) / (y2 - y1)
            t = np.clip(t, 0.0, 1.0)
            x_int = x1 + t * (x2 - x1)
            x_intersections.append(x_int)

            # Interpolate colour along the edge at this scanline
            p_i = np.array([x1, y1])
            p_j = np.array([x2, y2])
            c_int = vector_interp(p_i, p_j, vcolors[i], vcolors[j], y, dim=2)
            c_intersections.append(c_int)

        if len(x_intersections) < 2:
            continue

        # Left (A) and right (B) boundary points with their colours
        pairs = sorted(zip(x_intersections, c_intersections), key=lambda p: p[0])
        x_A, c_A = pairs[0]
        x_B, c_B = pairs[-1]

        p_A = np.array([x_A, y])
        p_B = np.array([x_B, y])

        x_start = max(0, int(np.ceil(x_A)))
        x_end = min(updated_img.shape[1] - 1, int(x_B))
        if x_start > x_end:
            continue

        # Interpolate colour horizontally from A to B for each pixel
        for x in range(x_start, x_end + 1):
            color = vector_interp(p_A, p_B, c_A, c_B, x, dim=1)
            updated_img[y, x] = color

    return updated_img


def shade_phong(v_pos: np.ndarray, v_nrm: np.ndarray, v_uvs: np.ndarray,
                tex: np.ndarray, cam_pos: np.ndarray, ka: float, kd: float,
                ks: float, n: float, l_pos, l_int, l_amb: np.ndarray,
                img: np.ndarray, bcoords: np.ndarray) -> np.ndarray:
    """
    Shade a single triangle with Phong shading.

    Args:
        v_pos   : (3, 2) projected pixel coords [x, y] of the 3 vertices
        v_nrm   : (3, 3) per-vertex unit normals (row i -> vertex i)
        v_uvs   : (3, 2) per-vertex texture coordinates
        tex     : (K, L, 3) texture image, float in [0, 1]
        cam_pos : (3,) camera position in WCS
        ka, kd, ks, n : Phong material coefficients
        l_pos, l_int : point light positions / intensities (N, 3)
        l_amb   : (3,) ambient light intensity
        img     : (res_h, res_w, 3) canvas with any pre-existing triangles
        bcoords : (3,) triangle barycenter in 3D (pre-projection), used to fix
                  the L and V directions of the lighting model

    Returns:
        updated_img : (res_h, res_w, 3) canvas with this triangle painted
    """
    updated_img = img.copy()

    vx = v_pos[:, 0]
    vy = v_pos[:, 1]

    ymin = max(0, int(np.min(vy)))
    ymax = min(updated_img.shape[0] - 1, int(np.max(vy)))

    edges = [(0, 1), (1, 2), (2, 0)]

    for y in range(ymin, ymax + 1):
        x_intersections = []
        n_intersections = []    # interpolated normal at each edge crossing
        uv_intersections = []   # interpolated uv at each edge crossing

        for (i, j) in edges:
            y1, y2 = vy[i], vy[j]
            x1, x2 = vx[i], vx[j]

            if y1 == y2:
                continue   # skip horizontal edges

            ek_min, ek_max = min(y1, y2), max(y1, y2)
            if y < ek_min or y > ek_max:
                continue
            if y == ek_max and ek_max != ymax:
                continue   # boundary convention (higher/right)

            t = (y - y1) / (y2 - y1)
            t = np.clip(t, 0.0, 1.0)
            x_int = x1 + t * (x2 - x1)
            x_intersections.append(x_int)

            # Interpolate normal and uv along the edge at this scanline
            p_i = np.array([x1, y1])
            p_j = np.array([x2, y2])
            n_int = vector_interp(p_i, p_j, v_nrm[i], v_nrm[j], y, dim=2)
            uv_int = vector_interp(p_i, p_j, v_uvs[i], v_uvs[j], y, dim=2)
            n_intersections.append(n_int)
            uv_intersections.append(uv_int)

        if len(x_intersections) < 2:
            continue

        # Left (A) and right (B) boundary points with their normal + uv
        pairs = sorted(zip(x_intersections, n_intersections, uv_intersections),
                       key=lambda p: p[0])
        x_A, n_A, uv_A = pairs[0]
        x_B, n_B, uv_B = pairs[-1]

        p_A = np.array([x_A, y])
        p_B = np.array([x_B, y])

        x_start = max(0, int(np.ceil(x_A)))
        x_end = min(updated_img.shape[1] - 1, int(x_B))
        if x_start > x_end:
            continue

        # Per-pixel: interpolate normal + uv, sample albedo, run the light model
        for x in range(x_start, x_end + 1):
            nrm_p = vector_interp(p_A, p_B, n_A, n_B, x, dim=1)
            uv_p = vector_interp(p_A, p_B, uv_A, uv_B, x, dim=1)

            # Re-normalise the interpolated normal (interpolation shrinks it)
            nlen = np.linalg.norm(nrm_p)
            if nlen > 1e-12:
                nrm_p = nrm_p / nlen

            albedo = _sample_texture(uv_p, tex)
            updated_img[y, x] = light(bcoords, nrm_p, albedo, cam_pos,
                                      ka, kd, ks, n, l_pos, l_int, l_amb)

    return updated_img
