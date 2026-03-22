import numpy as np
import cv2
import time
from rendering import render_img

print('=' * 50)
print('Demo: Texture Shading')
print('=' * 50)

start_time = time.time()

# Load input data from hw1.npy
print('\n1. Loading data from hw1.npy...')
data = np.load('hw1.npy', allow_pickle=True).item()

vertices = data['v_pos2d'].copy()      # (L, 2) 2D vertex positions
vertices[:, 1] = 511 - vertices[:, 1]  # Flip y-axis to match the expected orientation
vertices[:, 0] = 511 - vertices[:, 0]  # Flip x-axis to match the expected orientation
vcolors = data['v_clr']                # (L, 3) vertex RGB colors (not used for texture)
faces = data['t_pos_idx']              # (K, 3) triangle vertex indices
depth = data['depth']                  # (L,)   vertex depths
uvs = data['v_uvs'].copy()             # (L, 2) texture coordinates (u, v) normalized in [0, 1]
uvs[:, 1] = 1.0 - uvs[:, 1]            # Flip v to match flipped y axis

print(f'   Vertices : {vertices.shape}')
print(f'   Colors   : {vcolors.shape}')
print(f'   Faces    : {faces.shape}')
print(f'   Depth    : {depth.shape}')
print(f'   UVs      : {uvs.shape}')

# Load the texture image and convert from BGR to RGB and to float [0, 1]
print('\n2. Loading texture image...')
texImg_bgr = cv2.imread('loony-repeat.png')
texImg = cv2.cvtColor(texImg_bgr, cv2.COLOR_BGR2RGB).astype(float) / 255.0
print(f'   Texture size: {texImg.shape}')

# Render the image using texture shading
print('\n3. Rendering with texture shading...')
img = render_img(faces, vertices, vcolors, uvs, depth, shading='t', texImg=texImg)
end_time = time.time() - start_time
print(f'   Output image size: {img.shape}')
print(f'   Rendering time: {end_time:.2f} seconds')

# Convert from float [0, 1] RGB to uint8 [0, 255] BGR for OpenCV
img_bgr = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

# Save the output image
print('\n4. Saving output image...')
cv2.imwrite('output_texture.png', img_bgr)
print('   Saved: output_texture.png')

# Display the image
print('\n5. Displaying image...')
cv2.imshow('Texture Shading', img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

print('\n' + '=' * 50)