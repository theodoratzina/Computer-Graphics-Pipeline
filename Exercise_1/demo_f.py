import numpy as np
import cv2
import time
from rendering import render_img

print('=' * 50)
print('Demo: Flat Shading')
print('=' * 50)

start_time = time.time()

# Load input data from hw1.npy
print('\n1. Loading data from hw1.npy...')
data = np.load('hw1.npy', allow_pickle=True).item()

vertices = data['v_pos2d'].copy()      # (L, 2) 2D vertex positions
vertices[:, 1] = 511 - vertices[:, 1]  # Flip y-axis to match the expected orientation
vertices[:, 0] = 511 - vertices[:, 0]  # Flip x-axis to match the expected orientation
vcolors = data['v_clr']                # (L, 3) vertex RGB colors
faces = data['t_pos_idx']              # (K, 3) triangle vertex indices
depth = data['depth']                  # (L,)   vertex depths
uvs = data['v_uvs']                    # (L, 2) texture coordinates (not used for flat)

print(f'   Vertices : {vertices.shape}')
print(f'   Colors   : {vcolors.shape}')
print(f'   Faces    : {faces.shape}')
print(f'   Depth    : {depth.shape}')

# Render the image using flat shading
print('\n2. Rendering with flat shading...')
img = render_img(faces, vertices, vcolors, uvs, depth, shading='f')
end_time = time.time() - start_time
print(f'   Output image size: {img.shape}')
print(f'   Rendering time: {end_time:.2f} seconds')

# Convert from float [0, 1] RGB to uint8 [0, 255] BGR for OpenCV
img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Save the output image
print('\n3. Saving output image...')
cv2.imwrite('output_flat.png', img_bgr)
print('   Saved: output_flat.png')

# Display the image
print('\n4. Displaying image...')
cv2.imshow('Flat Shading', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('\n' + '=' * 50)
