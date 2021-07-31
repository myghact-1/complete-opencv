# https://docs.opencv.org/master/d5/d0f/tutorial_py_gradients.html

import cv2
import numpy as np

image  = cv2.imread("images\\sudoku.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
# ---------------------------------------------------------------------------- #
#* OpenCV provides three types of gradient filters or High-pass filters, Sobel, Scharr and Laplacian.

#? Sobel / Scharr filter and Laplacian Filter

sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

blurred = cv2.GaussianBlur(gray, (5,5), 0) # check without blurring
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)

show_image("Original image", gray)
show_image("Sobel X", sobelx)
show_image("Sobel Y", sobely)
show_image("Laplacian Image", laplacian, destroy=True)

# ---------------------------------------------------------------------------- #
#? Important

abs_sobelx = np.abs(sobelx)
final_res = np.uint8(abs_sobelx)
show_image("Abs Sobelx", final_res, destroy=True)