import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image_path = "files/tuning_images/extracted_frame.jpg"
image = cv2.imread(image_path)

def add_grid(image, grid_size=50, color=(255, 0, 0), thickness=1):
    img_with_grid = image.copy()
    h, w = img_with_grid.shape[:2]
    for y in range(0, h, grid_size):
        cv2.line(img_with_grid, (0, y), (w, y), color, thickness)
    for x in range(0, w, grid_size):
        cv2.line(img_with_grid, (x, 0), (x, h), color, thickness)
    return img_with_grid

# Adjusted distortion coefficients for desired effects
k1, k2, p1, p2, k3 = -0.4, 0.15, 0.0, 0.0, 0.15  # Coefficients for pinching the middle

# Get the dimensions of the image
h, w = image.shape[:2]

# Camera matrix
camera_matrix = np.array([[w, 0, w / 2],
                          [0, h, h / 2],
                          [0, 0, 1]], dtype=np.float32)

# Distortion coefficients
dist_coeffs = np.array([k1, k2, p1, p2, k3], dtype=np.float32)

# Undistort the image
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
undistorted_image = cv2.undistort(image, camera_matrix, dist_coeffs, None, new_camera_matrix)

# Add grid to images
image_with_grid = add_grid(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
undistorted_image_with_grid = add_grid(cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2RGB))

# Display the original and undistorted images with grid
plt.figure(figsize=(20, 40))

plt.subplot(2, 1, 1)
plt.title('Original Image with Grid')
plt.imshow(image_with_grid)

plt.subplot(2, 1, 2)
plt.title('Undistorted Image with Grid (Cropped)')
plt.imshow(undistorted_image_with_grid)

plt.show()