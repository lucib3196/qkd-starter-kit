import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the dictionary we want to use
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Generate a marker
marker_id = [1,5,10,42,20]
marker_size = 200  # Size in pixels
for marker in marker_id:
    print(marker)
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker, marker_size)
    cv2.imwrite(f'marker_{marker}.png', marker_image)
    # plt.imshow(marker_image, cmap='gray', interpolation='nearest')
    # plt.axis('off')  # Hide axes
    # plt.title(f'ArUco Marker {marker_id}')
    # plt.show()