import numpy as np
from shading import f_shading, g_shading, t_shading


def render_img(faces, vertices, vcolors, uvs, depth, shading, texImg=None):
    """
    Render a 2D image of a 3D object by filling K triangles using the specified 
    shading method. Triangles are painted back-to-front (painter's algorithm) based 
    on their mean vertex depth, so closer triangles correctly overlap farther ones.

    Args:
        faces    : (K, 3) int array, vertex indices of each triangle
        vertices : (L, 2) int array, 2D coordinates of each vertex
        vcolors  : (L, 3) float array, RGB color of each vertex in [0, 1]
        uvs      : (L, 2) float array, texture coordinates (u, v) normalized in [0, 1]
        depth    : (L,) float array, depth (z) of each vertex before projection
        shading  : str, shading mode — "f" for flat, "g" for Gouraud, "t" for texture
        texImg   : texture image (required if shading == "t")

    Returns:
        img : (M, N, 3) float array, rendered image with white background
    """
    # Set canvas size
    M, N = 512, 512

    # Initialize canvas with white background
    img = np.ones((M, N, 3), dtype=float)

    # Compute the depth of each triangle as the mean of its vertices' depths
    triangle_depths = np.mean(depth[faces], axis=1)

    # Sort triangles by depth in descending order (farthest first)
    sorted_indices = np.argsort(triangle_depths)[::-1]

    for idx in sorted_indices:
        # Extract the three vertex indices of this triangle
        face = faces[idx]

        # Gather the 2D coordinates and colors of the three vertices
        tri_vertices = vertices[face]  # (3, 2)
        tri_vcolors = vcolors[face]    # (3, 3)

        # Call the appropriate shading routine
        if shading == "f":
            img = f_shading(img, tri_vertices, tri_vcolors)

        elif shading == "g":
            img = g_shading(img, tri_vertices, tri_vcolors)

        elif shading == "t":
            tri_uvs = uvs[face]  # (3, 2)
            img = t_shading(img, tri_vertices, tri_uvs, texImg)

    return img