# https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html

# ---------------------------------------------------------------------------- #
#* types of distortion caused by cameras
#* how to find the intrinsic and extrinsic properties of a camera
#* how to undistort images based off these properties
# ---------------------------------------------------------------------------- #

import cv2
import numpy as np
from skimage import data

# chess = 