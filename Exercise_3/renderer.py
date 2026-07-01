import numpy as np
from camera import lookat, perspective_project, rasterize
from shading import calc_normals, shade_gouraud, shade_phong


def render_object(v_pos: np.ndarray, v_uvs: np.ndarray, t_pos_idx: np.ndarray,
                  tex: np.ndarray, plane_h: int, plane_w: int, res_h: int,
                  res_w: int, focal: float, eye: np.ndarray, up: np.ndarray,
                  target: np.ndarray, ka: float, kd: float, ks: float, n: float,
                  l_pos, l_int, l_amb: np.ndarray, shader: str) -> np.ndarray:
    """
    Render a 3-D object into a colour image using a Phong material and one or
    more point light sources.

    Pipeline:
      1. per-vertex normals via calc_normals (in WCS, before projection)
      2. perspective projection of the vertices onto the camera plane (hw2)
      3. rasterization to pixel coordinates (y flipped so row 0 is the top)
      4. triangles having any vertex outside the camera plane are skipped
      5. painter's algorithm: triangles painted back-to-front by mean depth,
         each filled by the selected shader (gouraud / phong)

    For every triangle the light/view directions are fixed by the 3-D
    barycenter (computed before projection) and passed to the shader.

    Args:
        v_pos     : (3, Nv) vertex coordinates in WCS
        v_uvs     : (Nv, 2) per-vertex texture coordinates in [0, 1]
        t_pos_idx : (NT, 3) triangle vertex indices (0-based)
        tex       : (K, L, 3) texture image, float in [0, 1]
        plane_h, plane_w : camera plane size (world units)
        res_h, res_w : output image resolution (pixels)
        focal  : focal length (camera-plane distance)
        eye    : (3,) camera centre in WCS
        up     : (3,) camera up vector
        target : (3,) camera look-at point
        ka, kd, ks, n : Phong material coefficients
        l_pos, l_int : point light positions / intensities (N, 3)
        l_amb  : (3,) ambient light intensity
        shader : "gouraud" or "phong"

    Returns:
        img : (res_h, res_w, 3) rendered image with white background
    """
    # Normalize input layouts (spec vs data discrepancies)
    v_pos = np.asarray(v_pos, dtype=float)
    if v_pos.shape[0] != 3 and v_pos.shape[1] == 3:
        v_pos = v_pos.T   # -> (3, Nv)

    v_uvs = np.asarray(v_uvs, dtype=float)
    if v_uvs.shape[1] != 2 and v_uvs.shape[0] == 2:
        v_uvs = v_uvs.T   # -> (Nv, 2)

    faces = np.asarray(t_pos_idx)
    if faces.ndim == 2 and faces.shape[1] != 3 and faces.shape[0] == 3:
        faces = faces.T   # -> (NT, 3)
    faces = faces.astype(int)

    eye = np.asarray(eye, dtype=float).reshape(3)
    up = np.asarray(up, dtype=float).reshape(3)
    target = np.asarray(target, dtype=float).reshape(3)

    # Step 1: per-vertex normals (WCS, before projection)
    normals = calc_normals(v_pos, faces)   # (3, Nv)

    # Step 2: project vertices onto the camera plane
    R, t = lookat(eye, up, target)
    pts_2d, depth = perspective_project(v_pos, focal, R, t)

    # Step 3: rasterize to pixels, flip y so row 0 is the top
    pix = rasterize(pts_2d, plane_w, plane_h, res_w, res_h)
    px = pix[0]
    py_img = (res_h - 1) - pix[1]
    verts_px = np.stack([px, py_img], axis=1).astype(int)   # (Nv, 2)

    # White canvas
    img = np.ones((res_h, res_w, 3), dtype=float)

    # Painter's algorithm: farthest triangles first
    tri_depth = np.mean(depth[faces], axis=1)
    order = np.argsort(tri_depth)[::-1]

    P = v_pos.T     # (Nv, 3) world positions, for barycenters
    N = normals.T   # (Nv, 3) per-vertex normals

    for idx in order:
        f = faces[idx]
        vp = verts_px[f]   # (3, 2) pixel coords

        # step 4: clip — skip triangles with any vertex outside the plane,
        # or with any vertex at/behind the camera (non-positive depth)
        if (np.any(vp[:, 0] < 0) or np.any(vp[:, 0] >= res_w) or
                np.any(vp[:, 1] < 0) or np.any(vp[:, 1] >= res_h)):
            continue
        if np.any(depth[f] <= 0):
            continue

        tri_n = N[f]                  # (3, 3) vertex normals
        tri_uv = v_uvs[f]             # (3, 2)
        bcoords = P[f].mean(axis=0)   # (3,) 3-D barycenter

        # step 5: fill with the selected shader
        if shader == "phong":
            img = shade_phong(vp, tri_n, tri_uv, tex, eye, ka, kd, ks, n,
                              l_pos, l_int, l_amb, img, bcoords)
        else:   # default / "gouraud"
            img = shade_gouraud(vp, tri_n, tri_uv, tex, eye, ka, kd, ks, n,
                                l_pos, l_int, l_amb, img, bcoords)

    return img
