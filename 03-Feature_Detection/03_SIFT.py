

import cv2
import numpy as np

# https://www.pexels.com/@expect-best-79873
image = cv2.imread("images\\building.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

sift = cv2.SIFT_create()
kp = sift.detect(gray, None)
descriptor = sift.compute(gray, kp)

# print('descriptor: ', descriptor)
####---OR---####
# kp, des = sift.detectAndCompute(gray, None)

image = cv2.drawKeypoints(gray, kp, image, 
                          flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

cv2.imshow("Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()