# https://docs.opencv.org/master/d3/d05/tutorial_py_table_of_contents_contours.html

import cv2
import numpy as np

image  = cv2.imread("images\\cog2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

image2  = cv2.imread("images\\cog.jpg")
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw contours
cnt = contours[0]
result = cv2.drawContours(image.copy(), [cnt], 0, (0,255,0), 2)

show_image("Image with contours", result)

#? Contour Approximation
epsilon = 0.1 * cv2.arcLength(cnt, True)
approx = cv2.approxPolyDP(cnt, epsilon, True)

result_2 = cv2.drawContours(image.copy(), [approx], 0, (0,255,0), 2)

show_image("Image with approx contours", result_2, destroy=True)

# ---------------------------------------------------------------------------- #
#? Convex Hull

_, thresh = cv2.threshold(gray2, 150, 255, cv2.THRESH_BINARY_INV)

# find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw contours
cnt = contours[0]
result = cv2.drawContours(image2.copy(), [cnt], 0, (0,255,0), 2)

show_image("Image 2 with contours", result)

hull = cv2.convexHull(cnt)

result_2 = cv2.drawContours(image2.copy(), [approx], 0, (0,0,255), 2)

show_image("Image 2 with convex hull contours", result_2, destroy=True)

# ---------------------------------------------------------------------------- #