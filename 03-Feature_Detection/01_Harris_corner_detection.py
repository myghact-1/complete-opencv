
# https://docs.opencv.org/master/dc/d0d/tutorial_py_features_harris.html

import cv2
import numpy as np

image = cv2.imread("images\\shapes.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray32 = np.float32(gray)

# ---------------------------------------------------------------------------- #
#? Harris Corner Detection

# detect the corners
dst = cv2.cornerHarris(gray32, 2, 3, 0.04)
cv2.imshow("Corners", dst)
cv2.waitKey(0)

# dilate the detcted corners
corners = cv2.dilate(dst, None)
cv2.imshow("Dilated Corners", corners)
cv2.waitKey(0)

##--- Show the corners on image
# Threshold for an optimal image
image[corners>0.01*corners.max()] = [0,0,255]
cv2.imshow("Final output", image)
cv2.waitKey(0)

cv2.destroyAllWindows()
exit()
# ---------------------------------------------------------------------------- #
#? Corner with SubPixel Accuracy

corners = cv2.cornerHarris(gray32, 2, 3, 0.04)
# dilate the detected corners
corners = cv2.dilate(corners, None)
# threshold
_, corners = cv2.threshold(corners, 0.01*corners.max(), 255, 0)

corners = np.uint8(corners)

# find centroids

ret, labels, stats, centroids = cv2.connectedComponentsWithStats(corners)

# define the criteria to stop and refine the corners
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001)

corners = cv2.cornerSubPix(gray, np.float32(centroids), (5,5), (-1,-1), criteria)

# Draw the result

res = np.hstack((centroids, corners))
res = np.int0(res)
image[res[:,1], res[:,0]] = [255, 0, 0]
image[res[:,3], res[:,2]] = [0, 255, 0]

cv2.imshow("Final output", image)
cv2.waitKey(0)

cv2.destroyAllWindows()