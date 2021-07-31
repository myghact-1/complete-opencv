# https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html

import cv2
import numpy as np


image = cv2.imread("images\\shapes.jpg")

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Average Blurring

show_image("Original Image", image)
kernels = [(3,3), (5,5), (7,7), (9,9)]
for kernel in kernels:
    blurred = cv2.blur(image, kernel)
    show_image(f"Blurred Image with kernel {kernel}", blurred)

    
#? Gaussian Blurring

kernels = [(3,3), (5,5), (7,7), (9,9)]
for kernel in kernels:
    gaussian_blur = cv2.GaussianBlur(image, kernel, 0)
    show_image(f"Gaussian Blur Image with kernel {kernel}", gaussian_blur)
 
    
#? Median Blurring (salt-and-pepper noise removal)

median = cv2.medianBlur(image, 3) # change the kernel size
show_image("Median Blurred Image", median, destroy=True)


#? Bilateral Filter (Keeps the edges sharp)

bilateral = cv2.bilateralFilter(image, 7, 75, 75)
show_image("Bilateral filter Image", bilateral, destroy=True)


#? Custom image filtering

kernel = np.ones(shape=(5,5), dtype=np.uint8) / 25
custom_blurred = cv2.filter2D(image, -1, kernel)
show_image("Custom blurred image", custom_blurred, destroy=True)