# https://docs.opencv.org/master/d1/d89/tutorial_py_orb.html

import cv2
import numpy as np

image = cv2.imread("images\\building.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize the ORB
orb = cv2.ORB_create(nfeatures=100)

# find the keypoints
kp = orb.detect(gray, None)

# compute the descriptors
kp, des = orb.compute(gray, kp)

# draw keypoints
image2 = cv2.drawKeypoints(gray, kp, image.copy())

cv2.imshow("ORB Kp", image2)
cv2.waitKey(0)
cv2.destroyAllWindows()