import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
from transforms import Trafo
from renderer import render_object
matplotlib.use("Agg")

"""
Static camera, object performs one full clockwise 
rotation around the world Y axis in 1 second

Output: 25 PNG frames saved in the folder 'demo1_frames/'
"""
# 1. Load scene data
data = np.load("data.npy", allow_pickle=True).item()

# Vertex positions: stored as (3, N)
v_pos = data["v_pos"]

# UV texture coordinates (N, 2)
v_uvs = data["v_uvs"]

# Face indices (F, 3)
t_pos_idx = data["t_pos_idx"]

# Camera intrinsics
focal = float(data["k_f"])                 # focal length
plane_h = float(data["k_sensor_height"])   # camera plane height (world units)
plane_w = float(data["k_sensor_width"])    # camera plane width  (world units)

# Camera extrinsics — stored as (3,1), flatten to (3,)
eye = data["k_cam_eye"].flatten()         # camera centre in WCS
up = data["k_cam_up"].flatten()           # up vector
target = data["k_cam_target"].flatten()   # look-at point

# Animation parameters
n_frames = int(data["k_fps"] * data["k_duration"])   # 25 frames

# Texture image (float [0,1])
texImg = np.array(Image.open("loony-repeat.png").convert("RGB")).astype(float) / 255.0
texImg = np.flipud(texImg)

# Image resolution
res_h, res_w = 512, 512

# Rotation axis and centre
y_axis = np.array([0.0, 1.0, 0.0])
rot_center = np.zeros(3)

# Output directory
out_dir = "demo1_frames"
os.makedirs(out_dir, exist_ok=True)

print(f"[demo1] {n_frames} frames | focal = {focal} | plane = {plane_h} x {plane_w}")
print(f"[demo1] eye = {eye} | target = {target}")

# 2. Render loop
for frame in range(n_frames):

    # Positive angle is passed to rotate(), which internally applies CW rotation
    angle = 2.0 * np.pi * frame / n_frames

    # Fresh Trafo each frame — avoids accumulation across frames
    tr = Trafo()
    tr.rotate(y_axis, angle, rot_center)

    # Rotate vertices: xform_pnts returns (3, N) — transpose to (N, 3) for render_object
    v_pos_rot = tr.xform_pnts(v_pos).T

    # Render
    img = render_object(
        v_pos = v_pos_rot,
        v_clr = np.zeros((v_pos_rot.shape[0], 3)),   # unused (texture mode)
        t_pos_idx = t_pos_idx,
        plane_h = plane_h,
        plane_w = plane_w,
        res_h = res_h,
        res_w = res_w,
        focal = focal,
        eye = eye,
        up = up,
        target = target,
        v_uvs = v_uvs,
        texImg = texImg
    )

    # Save frame
    fname = os.path.join(out_dir, f"demo1_frame_{frame:02d}.png")
    plt.imsave(fname, np.clip(img, 0.0, 1.0))
    print(f"frame {frame + 1:2d}/{n_frames}: angle = {np.degrees(angle):6.1f} -> {fname}")

print(f"\n[demo1] Done. Frames saved in '{out_dir}/'")
