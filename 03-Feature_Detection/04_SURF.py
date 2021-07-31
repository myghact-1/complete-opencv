# https://docs.opencv.org/master/df/dd2/tutorial_py_surf_intro.html

import cv2
import numpy as np

# https://www.pexels.com/@expect-best-79873
image = cv2.imread("images\\building.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
descriptor = sift.compute(gray, kp)

# ---------------------------------------------------------------------------- #
#? SURF

# surf = cv2