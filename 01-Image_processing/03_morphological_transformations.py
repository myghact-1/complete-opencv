# https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html


import cv2
import numpy as np

img1 = cv2.imread("images\\opencv1.jpg", 0) # gray scale image
img2 = cv2.imread("images\\opencv2.jpg", 0) # gray scale image
img3 = cv2.imread("images\\opencv3.jpg", 0) # gray scale image

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Erosion & Dilation
kernel = np.ones(shape=(5,5), dtype=np.uint8)
erode = cv2.erode(img1, kernel, iterations=1)
dilation = cv2.dilate(img1, kernel, iterations=1)

show_image("Original Image 1", img1)
show_image("Erosion", erode)
show_image("Dilation", dilation)

#? Gradient, top hat, and black hat
gradient = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel, iterations=1)
top_hat = cv2.morphologyEx(img1, cv2.MORPH_TOPHAT, kernel, iterations=1)
black_hat = cv2.morphologyEx(img1, cv2.MORPH_BLACKHAT, kernel, iterations=1)
cross = cv2.morphologyEx(img1, cv2.MORPH_CROSS, kernel, iterations=1)
ellipse = cv2.morphologyEx(img1, cv2.MORPH_ELLIPSE, kernel, iterations=1)

show_image("gradient", gradient)
show_image("top_hat", top_hat)
show_image("black_hat", black_hat)
show_image("cross", cross)
show_image("ellipse", ellipse, destroy=True)

# ---------------------------------------------------------------------------- #
#? Opening (erosion followed by dilation)--> (Removes foreground noise)

opening = cv2.morphologyEx(img2, cv2.MORPH_OPEN, kernel, iterations=1)
show_image("Original Image with foreground noise", img2)
show_image("Opening", opening, destroy=True)

# ---------------------------------------------------------------------------- #
#? Closing (dilation followed by erosion)--> (Removes background noise)

closing = cv2.morphologyEx(img3, cv2.MORPH_CLOSE, kernel, iterations=1)
show_image("Original Image with background noise", img3)
show_image("Closing", closing, destroy=True)