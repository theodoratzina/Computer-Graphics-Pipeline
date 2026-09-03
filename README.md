# 🎨 Computer Graphics Pipeline

A complete, from-scratch 3D software rendering engine built in Python, developed for the course *Computer Graphics* (2025–2026) at the Aristotle University of Thessaloniki, Department of Electrical and Computer Engineering.

This repository implements a full graphics rendering pipeline without relying on external graphics APIs. Using only NumPy for matrix mathematics and vectorization, the engine progresses from basic 2D triangle rasterization to a full 3D perspective projection system with dynamic Phong illumination.

---

## 📋 Table of Contents

- [Project Structure](#project-structure)
- [Exercise 1 — Triangle Rasterization & Shading](#exercise-1--triangle-rasterization--shading)
- [Exercise 2 — 3D Transformations & Camera Model](#exercise-2--3d-transformations--camera-model)
- [Exercise 3 — Illumination & View](#exercise-3--illumination--view)
- [Results & Key Highlights](#results--key-highlights)
- [Installation](#installation)
- [Usage](#usage)

---

## Project Structure

```text
.
├── Exercise_1/
│   ├── figures/
│   ├── demo_f.py
│   ├── demo_g.py
│   ├── demo_t.py
│   ├── hw1.npy
│   ├── hw1_2026.pdf
│   ├── loony-repeat.png
│   ├── rendering.py
│   ├── report_1.pdf
│   ├── shading.py
│   └── utils.py
│
├── Exercise_2/
│   ├── demo1_frames/
│   ├── demo2_frames/
│   ├── camera.py
│   ├── data.npy
│   ├── demo1.py
│   ├── demo2.py
│   ├── hw2_2026.pdf
│   ├── loony-repeat.png
│   ├── renderer.py
│   ├── rendering.py
│   ├── report_2.pdf
│   ├── shading.py
│   ├── sphere.obj
│   ├── transforms.py
│   └── utils.py
│
├── Exercise_3/
│   ├── renders/
│   ├── renders_back/
│   ├── camera.py
│   ├── demo.py
│   ├── hw3.npy
│   ├── hw3_2026.pdf
│   ├── lighting.py
│   ├── loony-repeat.png
│   ├── renderer.py
│   ├── report_3.pdf
│   ├── shading.py
│   ├── sphere.obj
│   └── utils.py
└──
```

---

## Exercise 1 — Triangle Rasterization & Shading

Builds the foundational 2D rasterization algorithms for drawing and filling triangles on a canvas.
- **Scanline Algorithm:** Custom implementation of a triangle scanline fill algorithm relying heavily on linear interpolation for computing edge intersections and spans.
- **Shading Methods:**
  - **Flat Shading:** Assigns a single uniform color to each triangle based on the average of its vertices.
  - **Gouraud Shading:** Achieves smooth color transitions by bi-linearly interpolating vertex colors across the triangle surface.
- **Texture Mapping:** Maps a 2D image (`loony-repeat.png`) onto 3D geometry by interpolating normalized UV coordinates $(u,v) \in [0,1]^2$ and applying Nearest-Neighbor sampling.
- **Visibility:** Resolves occlusion using the **Painter's Algorithm**, sorting triangles by depth (calculated via barycenters) and rendering from back to front.

---

## Exercise 2 — 3D Transformations & Camera Model

Introduces spatial awareness by moving from 2D canvas coordinates to a fully projected 3D world.
- **Affine Transformations:** Implements a `Trafo` class to handle cumulative translations and rotations around arbitrary axes using Rodrigues' rotation formula.
- **Camera Extrinsics:** Computes the View Matrix via a `lookat` function, establishing an orthonormal basis (Camera Coordinate System) given an `eye`, `target`, and `up` vector.
- **Pinhole Camera Model:** Applies perspective projection to map 3D vertices to a 2D projection plane, actively computing point depths for later occlusion handling.
- **Rasterization:** Translates continuous projection plane coordinates into discrete pixel indices corresponding to the final image resolution.

---

## Exercise 3 — Illumination & View

Implements physical light interactions and advanced shading models to achieve photorealistic rendering.
- **Vertex Normals:** Dynamically calculates continuous surface normals using an area-weighted accumulation of cross-product face normals.
- **Phong Reflection Model:** Computes the radiant intensity at any given point by summing Ambient, Lambertian Diffuse, and Specular reflection components across multiple distinct point light sources.
- **Advanced Shaders:**
  - **Gouraud Shader:** Evaluates the full Phong lighting model only at the three vertices of a triangle and interpolates the resulting colors.
  - **Phong Shader:** Interpolates the 3D normal vectors and UV coordinates per-pixel, evaluating the full lighting and texture sampling individually for every pixel on the span.
- **Clipping:** Discards triangles lying entirely outside the camera plane or behind the camera ($z \le 0$) prior to rasterization.

---

## Results & Key Highlights

*   **Custom Vectorization:** Extensive use of NumPy broadcasting and matrix operations ensures that computationally heavy tasks (like span interpolations and texture sampling) run efficiently without native C/C++ backends.
*   **Gouraud vs. Phong Highlights:** The implementation clearly demonstrates the limitations of Gouraud shading. Specular highlights that fall entirely within a triangle face (and not near a vertex) are lost in Gouraud but accurately captured as sharp, pixel-perfect reflections in Phong shading.
*   **Dynamic Animation Engine:** Exercise 2 proves the mathematical stability of the system by generating smooth 25 FPS animations, cleanly rendering both a rotating object and an orbiting camera around a static scene without matrix accumulation drift.

<div align="center">
  <img width="45%" src="https://github.com/theodoratzina/Computer-Graphics-Pipeline/blob/main/Exercise_1/figures/output_texture.png" hspace="2%" />
  <img width="45%" src="https://github.com/theodoratzina/Computer-Graphics-Pipeline/blob/main/Exercise_3/renders_back/phong_combined_back.png" />
  <br>
</div>

---

## Installation

### Requirements

- Python 3.10+
- NumPy
- OpenCV (`opencv-python`)
- Matplotlib

Install dependencies:
```bash
pip install numpy opencv-python matplotlib
```
---

## Usage
Navigate to the respective exercise directories and run the demo scripts:

```bash
# Exercise 1: Rasterization & Shading Models
cd Exercise_1
python demo_f.py  # Generates output_flat.png
python demo_g.py  # Generates output_gouraud.png
python demo_t.py  # Generates output_texture.png

# Exercise 2: Camera Orbit & Projections
cd Exercise_2
python demo1.py   # Generates frames for rotating object
python demo2.py   # Generates frames for orbiting camera

# Exercise 3: Full Illumination Pipeline
cd Exercise_3
python demo.py    # Renders all combinations of light sources and shaders
```
