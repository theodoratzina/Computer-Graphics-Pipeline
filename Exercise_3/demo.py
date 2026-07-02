import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from renderer import render_object
from shading import calc_normals
from camera import lookat, perspective_project

"""
Demo for hw3

Reads object, parameters from 'hw3.npy' and texture from 'loony-repeat.png', 
renders sphere for both shaders (Gouraud and Phong). For each shader:

  Component breakdown (all 3 lights, one lighting term at a time):
    <shader>_ambient.png    (only ka)
    <shader>_diffuse.png    (only kd)
    <shader>_specular.png   (only ks)
    <shader>_combined.png   (ka + kd + ks, all lights)

  Per-light-source breakdown (full material, one point light at a time):
    <shader>_light0.png
    <shader>_light1.png
    <shader>_light2.png
    <shader>_all_lights.png

All frames are written into the 'renders/' folder
"""

# 1. Load scene data
data = np.load("hw3.npy", allow_pickle=True).item()

v_pos = data["v_pos"]           # (3, Nv)
v_uvs = data["v_uvs"]           # (Nv, 2)
t_pos_idx = data["t_pos_idx"]   # (NT, 3)

plane_h = int(data["plane_h"])
plane_w = int(data["plane_w"])
res_h = int(data["res_h"])
res_w = int(data["res_w"])
focal = float(data["focal"])

eye = np.asarray(data["cam_pos"]).flatten()   # camera centre in WCS
up = np.asarray(data["up"]).flatten()
target = np.asarray(data["target"]).flatten()

ka = float(data["ka"])
kd = float(data["kd"])
ks = float(data["ks"])
n = float(data["n"])

l_pos = data["l_pos"]   # list of point light positions
l_int = data["l_int"]   # list of point light intensities
l_amb = np.asarray(data["l_amb"]).flatten()

# 2. Texture image: RGB float in [0, 1], flipped vertically 
#    so it aligns with the v texture axis
tex = np.array(Image.open("loony-repeat.png").convert("RGB")).astype(float) / 255.0
tex = np.flipud(np.fliplr(tex))

# 3. Output directory
out_dir = "renders"
os.makedirs(out_dir, exist_ok=True)

print(f"[demo] res = {res_h}x{res_w} | focal = {focal} | plane = {plane_h}x{plane_w}")
print(f"[demo] {len(l_pos)} point lights | ka={ka} kd={kd} ks={ks} n={n}")


def print_diagnostics():
    """Print quantitative sanity checks that justify the visual results."""
    print("\n" + "=" * 62)
    print("SCENE DIAGNOSTICS")
    print("=" * 62)
 
    # Geometry / npy consistency
    r = np.linalg.norm(v_pos, axis=0)
    print("\n[geometry]")
    print(f"  vertices Nv   : {v_pos.shape[1]}   triangles NT: {t_pos_idx.shape[0]}")
    print(f"  vertex radius : mean={r.mean():.4f} std={r.std():.5f}  (unit sphere at origin)")
    print(f"  uv range      : [{v_uvs.min():.2f}, {v_uvs.max():.2f}]  (normalized)")
    print(f"  camera eye={eye}  target={target}  up={up}")
 
    # Normals: outward & unit
    nrm = calc_normals(v_pos, t_pos_idx)             # (3, Nv)
    unit = np.allclose(np.linalg.norm(nrm, axis=0), 1.0, atol=1e-6)
    radial = v_pos / np.linalg.norm(v_pos, axis=0)
    outward = np.einsum('ij,ij->j', nrm, radial)
    print("\n[normals]")
    print(f"  all unit length  : {unit}")
    print(f"  outward fraction : {100*np.mean(outward > 0):.1f}% (dot with radial dir, mean={outward.mean():.4f})")
 
    # Which hemisphere is visible (painter's order)
    R, t = lookat(eye, up, target)
    _, depth = perspective_project(v_pos, focal, R, t)
    N = nrm.T
    P = v_pos.T
    tri_depth = np.mean(depth[t_pos_idx], axis=1)
    order = np.argsort(tri_depth)[::-1]   # far first, near last (on top)
    tri_z = np.mean(v_pos[2][t_pos_idx], axis=1)
    print("\n[visibility]  (painter's algorithm draws near side last = on top)")
    print(f"  triangles on top : mean world-z = {tri_z[order[-500:]].mean():+.3f} (near side, z<0 -> visible)")
    print(f"  triangles behind : mean world-z = {tri_z[order[:500]].mean():+.3f} (far side, hidden)")
 
    # Per-light diffuse energy delivered to the visible hemisphere
    view = eye[None, :] - P
    view = view / np.linalg.norm(view, axis=1, keepdims=True)
    visible = np.einsum('ij,ij->i', N, view) > 0
    print("\n[light contribution to the visible hemisphere]")
    print(f"  visible vertices : {visible.sum()} / {len(P)}")
    names = ['light0', 'light1', 'light2']
    for i, (lp, li) in enumerate(zip(l_pos, l_int)):
        L = np.asarray(lp)[None, :] - P
        L = L / np.linalg.norm(L, axis=1, keepdims=True)
        ndotl = np.clip(np.einsum('ij,ij->i', N, L), 0, None)
        lit_vis = (ndotl > 0) & visible
        print(f"  {names[i]} pos={np.asarray(lp)} int={np.asarray(li)}: "
              f"lit&visible={lit_vis.sum():6d}  total N.L={ndotl[lit_vis].sum():8.1f}")
    print("=" * 62 + "\n")
 
 
print_diagnostics()


def render_and_save(shader, k_a, k_d, k_s, lp, li, name):
    """Render one configuration and save it as a PNG in out_dir."""
    img = render_object(
        v_pos, v_uvs, t_pos_idx, tex,
        plane_h, plane_w, res_h, res_w, focal,
        eye, up, target,
        k_a, k_d, k_s, n, lp, li, l_amb, shader
    )
    path = os.path.join(out_dir, name)
    plt.imsave(path, np.clip(img, 0.0, 1.0))
    print(f"  saved {path}")
    return img


# 4. Render loop over both shaders
for shader in ("gouraud", "phong"):
    print(f"\n[{shader}] component breakdown (all lights)")
    render_and_save(shader, ka, 0.0, 0.0, l_pos, l_int, f"{shader}_ambient.png")
    render_and_save(shader, 0.0, kd, 0.0, l_pos, l_int, f"{shader}_diffuse.png")
    render_and_save(shader, 0.0, 0.0, ks, l_pos, l_int, f"{shader}_specular.png")
    combined = render_and_save(shader, ka, kd, ks, l_pos, l_int,
                               f"{shader}_combined.png")

    print(f"[{shader}] per-light breakdown (full material)")
    for i in range(len(l_pos)):
        render_and_save(shader, ka, kd, ks, [l_pos[i]], [l_int[i]],
                        f"{shader}_light{i}.png")

    # "all lights" is identical to the combined image — save it under both names
    plt.imsave(os.path.join(out_dir, f"{shader}_all_lights.png"),
               np.clip(combined, 0.0, 1.0))
    print(f"  saved {os.path.join(out_dir, f'{shader}_all_lights.png')}")

print(f"\n[demo] Done. Images saved in '{out_dir}/'")
