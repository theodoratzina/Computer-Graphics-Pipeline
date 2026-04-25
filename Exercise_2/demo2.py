"""
demo2.py
--------
Static object at the world origin. The camera performs one full clockwise
orbit around the world Y axis (viewed from above) in 1 second, always
pointing at the object centre.
Output: 25 PNG frames saved in 'demo2_frames/'.

Usage:
    python demo2.py

Expected files in the same directory:
    data.npy     - scene data (v_pos, v_uvs, t_pos_idx, camera params)
    diffuse.jpg  - texture image for the sphere

Camera orbit (clockwise when viewed from +Y):
    eye_x = radius * sin(angle)
    eye_y = constant (same height as the static camera in demo1)
    eye_z = radius * cos(angle)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from renderer import render_object


# ---------------------------------------------------------------------------
# 1. Load scene data
# ---------------------------------------------------------------------------
data = np.load("data.npy", allow_pickle=True).item()

# Vertex positions: stored as (3, N) — transpose to (N, 3) for render_object
v_pos     = data["v_pos"].T                          # (N, 3)  — static in demo2

# UV texture coordinates (N, 2)
v_uvs     = data["v_uvs"]                            # (N, 2)

# Face indices (F, 3)
t_pos_idx = data["t_pos_idx"]                        # (F, 3)

# Camera intrinsics
focal   = float(data["k_f"])
plane_h = float(data["k_sensor_height"])
plane_w = float(data["k_sensor_width"])

# Camera orbit radius (provided in data)
radius = float(data["k_cam_radius"])                 # 5.0 world units

# Camera height: keep the same Y as the static camera eye from demo1
eye_height = float(data["k_cam_eye"].flatten()[1])   # Y component

# Up vector and look-at target (object is at origin)
up     = np.array([0.0, 1.0, 0.0])
target = np.zeros(3)

# Animation parameters
n_frames = int(data["k_fps"] * data["k_duration"])  # 25 frames

# Texture image (float [0,1])
texImg = np.array(Image.open("diffuse.jpg")).astype(float) / 255.0

# Image resolution (fixed by assignment)
res_h, res_w = 512, 512

# Output directory
OUT_DIR = "demo2_frames"
os.makedirs(OUT_DIR, exist_ok=True)

print(f"[demo2] {n_frames} frames | focal={focal} | plane={plane_h}x{plane_w}")
print(f"[demo2] orbit radius={radius}  eye_height={eye_height}")

# ---------------------------------------------------------------------------
# 2. Render loop
# ---------------------------------------------------------------------------
for frame in range(n_frames):

    # Clockwise orbit angle (viewed from +Y)
    angle = 2.0 * np.pi * frame / n_frames

    # Camera position on the orbit circle
    eye = np.array([
        radius * np.sin(angle),   # X
        eye_height,               # Y  (constant)
        radius * np.cos(angle),   # Z
    ])

    # Render (v_pos unchanged)
    img = render_object(
        v_pos     = v_pos,
        v_clr     = np.zeros((v_pos.shape[0], 3)),   # unused (texture mode)
        t_pos_idx = t_pos_idx,
        plane_h   = plane_h,
        plane_w   = plane_w,
        res_h     = res_h,
        res_w     = res_w,
        focal     = focal,
        eye       = eye,
        up        = up,
        target    = target,
        v_uvs     = v_uvs,
        texImg    = texImg,
    )

    # Save frame
    fname = os.path.join(OUT_DIR, f"demo2_frame_{frame:02d}.png")
    plt.imsave(fname, np.clip(img, 0.0, 1.0))
    print(f"  frame {frame + 1:2d}/{n_frames}  eye={np.round(eye,2)}  -> {fname}")

print(f"\n[demo2] Done. Frames saved in '{OUT_DIR}/'")
