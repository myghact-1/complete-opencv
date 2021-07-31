# https://docs.opencv.org/master/df/d0c/tutorial_py_fast.html

import cv2
import numpy as np

image = cv2.imread("images\\building.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# initialize the fast algorithm
fast = cv2.FastFeatureDetector_create()
# detect key-points
kp = fast.detect(gray, None)
# draw keypoints
image2 = cv2.drawKeypoints(gray, kp, image.copy())

# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "Total Keypoints with nonmaxSuppression: {}".format(len(kp)) )

cv2.imshow("Image with features", image2)
cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------------------------------------------------------------------------- #

# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(gray,None)

print( "Total Keypoints without nonmaxSuppression: {}".format(len(kp)) )

image3 = cv2.drawKeypoints(gray, kp, image.copy())
cv2.imshow("Image with features withot nonmaxsuppression", image3)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------------------- #