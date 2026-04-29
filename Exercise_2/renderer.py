import numpy as np
from typing import Optional
from camera import lookat, perspective_project, rasterize
from rendering import render_img


def render_object(v_pos: np.ndarray, v_clr: np.ndarray, t_pos_idx: np.ndarray, plane_h: float, 
                  plane_w: float, res_h: int, res_w: int, focal: float, eye: np.ndarray, 
                  up: np.ndarray, target: np.ndarray, v_uvs: Optional[np.ndarray] = None, 
                  texImg: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Renders a 3-D object from a given camera viewpoint

    Args:
        v_pos: (N, 3) 3-D vertex positions in WCS
        v_clr: (N, 3) per-vertex RGB colors in [0, 1]
        t_pos_idx: (F, 3) triangle face indices (0-based)
        plane_h: height of the camera plane (world units)
        plane_w: width of the camera plane (world units)
        res_h: image height in pixels
        res_w: image width in pixels
        focal: focal length (camera-plane distance from camera centre)
        eye: (3,) camera centre in WCS
        up: (3,) approximate up direction
        target: (3,) point the camera looks at
        v_uvs: (N, 2) optional UV texture coordinates in [0, 1]
        texImg: (H, W, 3) optional texture image float array in [0, 1]

    Returns:
        img: (res_h, res_w, 3) float array — rendered image (white background)
    """
    # Step 1 – camera parameters
    R, t = lookat(eye, up, target)

    # Step 2 – project 3-D vertices onto the camera plane
    pts_world = v_pos.T
    pts_2d, depth = perspective_project(pts_world, focal, R, t)

    # Step 3 – rasterize to pixel coordinates (y = 0 at bottom)
    pix = rasterize(pts_2d, plane_w, plane_h, res_w, res_h)

    px = pix[0]   # horizontal pixel index, 0 = left
    py = pix[1]   # vertical pixel index, 0 = bottom

    # Step 4 – flip y so that row 0 is the top of the image array
    py_img = (res_h - 1) - py

    # Build the (N, 2) integer vertex array expected by render_img: [x, y]
    vertices_2d = np.stack([px, py_img], axis=1).astype(int)

    # Step 5 – render using HW1 shading routines
    if v_uvs is not None and texImg is not None:
        shading = "t"
        uvs = v_uvs
    else:
        shading = "g"
        uvs = np.zeros((v_pos.shape[0], 2))   # placeholder — not used by g_shading

    img = render_img(
        faces = t_pos_idx,
        vertices = vertices_2d,
        vcolors = v_clr,
        uvs = uvs,
        depth = depth,
        shading = shading,
        texImg = texImg
    )

    return img
