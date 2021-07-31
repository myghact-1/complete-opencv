

import cv2
import numpy as np
from matplotlib import cm

image = cv2.imread("images\\desert.jpg")

image_copy = image.copy()

marker_image = np.zeros(image.shape[:2], dtype=np.int32)

segments = np.zeros(image.shape, dtype=np.uint8)

def create_rgb(i):
    return tuple(np.array(cm.tab10(i)[:3])*255)

colors = []

for i in range(10):
    colors.append(create_rgb(i))

# print(colors)

### Global variable
n_markers = 10
current_marker = 1
marker_update = False

### Mouse Callback
def mouse_callback(event, x, y, flags, param):
    global marker_update
    
    if event==cv2.EVENT_LBUTTONDOWN:
        # marker passed to the watershed algorithm
        cv2.circle(marker_image, (x, y), 10, (current_marker), -1)
        
        # user sees on the image
        cv2.circle(image_copy, (x,y), 10, colors[current_marker], -1)
        
        marker_update = True
        
        
cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

while True:
    
    cv2.imshow("Watershed Segments", segments)
    cv2.imshow("Image", image_copy)
    
    # break the while loop
    k = cv2.waitKey(1)
    if k==ord('q'):
        break
    
    # clearing all colors
    elif k==ord('c'):
        image_copy = image.copy()
        marker_image = np.zeros(image.shape[:2], dtype=np.int32)
        segments = np.zeros(image.shape, dtype=np.uint8)
        
    # update color choice
    elif k>0 and chr(k).isdigit():
        current_marker = int(chr(k))
        
    # update the markings
    if marker_update:
        
        marker_image_copy = marker_image.copy()
        
        cv2.watershed(image, marker_image_copy)
        
        segments = np.zeros(image.shape, dtype=np.uint8)
        
        for color_idx in range(n_markers):
            # coloring segments
            segments[marker_image_copy==(color_idx)] = colors[color_idx]
            
