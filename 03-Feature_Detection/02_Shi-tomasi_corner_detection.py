# https://docs.opencv.org/master/d4/d8c/tutorial_py_shi_tomasi.html

import cv2
import numpy as np

image = cv2.imread("images\\shapes.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

corners = cv2.goodFeaturesToTrack(gray, 50, 0.04, 10)
corners = np.int0(corners)

for i in corners:
    x, y = i.ravel()
    cv2.circle(image, (x, y), 3, (0,255,0), -1)
    
cv2.imshow("Corners", image)
cv2.waitKey(0)
cv2.destroyAllWindows()