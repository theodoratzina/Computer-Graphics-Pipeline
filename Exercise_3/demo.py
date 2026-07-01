import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from renderer import render_object

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
tex = np.flipud(tex)

# 3. Output directory
out_dir = "renders"
os.makedirs(out_dir, exist_ok=True)

print(f"[demo] res = {res_h}x{res_w} | focal = {focal} | plane = {plane_h}x{plane_w}")
print(f"[demo] {len(l_pos)} point lights | ka={ka} kd={kd} ks={ks} n={n}")


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
