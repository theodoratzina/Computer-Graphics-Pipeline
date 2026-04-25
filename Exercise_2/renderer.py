import numpy as np
from typing import Optional

from transforms import lookat
from camera import perspective_project, rasterize
from rendering import render_img


def render_object(
    v_pos: np.ndarray,
    v_clr: np.ndarray,
    t_pos_idx: np.ndarray,
    plane_h: float,
    plane_w: float,
    res_h: int,
    res_w: int,
    focal: float,
    eye: np.ndarray,
    up: np.ndarray,
    target: np.ndarray,
    v_uvs: Optional[np.ndarray] = None,
    texImg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Renders a 3-D object from a given camera viewpoint.

    Full pipeline:
        1. lookat        : compute camera rotation R and translation t
        2. perspective_project : project 3-D world points → 2-D camera-plane coords
        3. rasterize     : map camera-plane coords → integer pixel coords
        4. y-flip        : convert from bottom-left origin to top-left (image array) origin
        5. render_img    : fill triangles with Gouraud or texture shading (HW1)

    Shading selection:
        - If v_uvs and texImg are provided → texture shading ("t")
        - Otherwise                        → Gouraud shading ("g")

    Args:
        v_pos     : (N, 3)  3-D vertex positions in WCS.
        v_clr     : (N, 3)  per-vertex RGB colors in [0, 1].
        t_pos_idx : (F, 3)  triangle face indices (0-based).
        plane_h   : height of the camera plane (world units).
        plane_w   : width  of the camera plane (world units).
        res_h     : image height in pixels.
        res_w     : image width  in pixels.
        focal     : focal length (camera-plane distance from camera centre).
        eye       : (3,) camera centre in WCS.
        up        : (3,) approximate up direction.
        target    : (3,) point the camera looks at.
        v_uvs     : (N, 2) optional UV texture coordinates in [0, 1].
        texImg    : (H, W, 3) optional texture image float array in [0, 1].

    Returns:
        img : (res_h, res_w, 3) float array — rendered image (white background).
    """
    # ------------------------------------------------------------------
    # Step 1 – camera parameters
    # ------------------------------------------------------------------
    R, t = lookat(eye, up, target)

    # ------------------------------------------------------------------
    # Step 2 – project 3-D vertices onto the camera plane
    # ------------------------------------------------------------------
    # perspective_project expects pts as (3, N), v_pos is (N, 3)
    pts_world = v_pos.T                              # (3, N)
    pts_2d, depths = perspective_project(pts_world, focal, R, t)
    # pts_2d : (2, N),  depths : (N,)

    # ------------------------------------------------------------------
    # Step 3 – rasterize to pixel coordinates (y = 0 at bottom)
    # ------------------------------------------------------------------
    pix = rasterize(pts_2d, plane_w, plane_h, res_w, res_h)
    # pix : (2, N) — row 0 = px (x / col), row 1 = py (y, 0 = bottom)

    px = pix[0]   # (N,) horizontal pixel index, 0 = left
    py = pix[1]   # (N,) vertical   pixel index, 0 = bottom

    # ------------------------------------------------------------------
    # Step 4 – flip y so that row 0 is the TOP of the image array
    # ------------------------------------------------------------------
    py_img = (res_h - 1) - py        # (N,)

    # Build the (N, 2) integer vertex array expected by render_img: [x, y]
    vertices_2d = np.stack([px, py_img], axis=1).astype(int)  # (N, 2)

    # ------------------------------------------------------------------
    # Step 5 – render using HW1 shading routines
    # ------------------------------------------------------------------
    if v_uvs is not None and texImg is not None:
        shading = "t"
        uvs = v_uvs                        # (N, 2)
    else:
        shading = "g"
        uvs = np.zeros((v_pos.shape[0], 2))  # placeholder — not used by g_shading

    img = render_img(
        faces    = t_pos_idx,        # (F, 3)
        vertices = vertices_2d,      # (N, 2)  pixel coords, y = 0 at top
        vcolors  = v_clr,            # (N, 3)
        uvs      = uvs,              # (N, 2)
        depth    = depths,           # (N,)
        shading  = shading,
        texImg   = texImg,
    )

    return img
