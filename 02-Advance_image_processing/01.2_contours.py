# https://docs.opencv.org/master/d3/d05/tutorial_py_table_of_contents_contours.html

import cv2
import numpy as np

image  = cv2.imread("images\\one_shape.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Bounding Rectangle

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

# find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# draw contours
cnt = contours[0]
result = cv2.drawContours(image.copy(), [cnt], 0, (0,255,0), 2)

show_image("Image with contours", result)

#! Straight rectangle

x,y,w,h = cv2.boundingRect(cnt)
straight_rect = cv2.rectangle(image.copy(), (x,y), (x+w, y+h), (0,0,255),3)

show_image("Image with Staright rectangle", straight_rect)

#! Best fit rectangle

rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
best_rect = cv2.drawContours(image.copy(), [box], 0, (0,0,255), 2)

show_image("Image with Best fit rectangle", best_rect)

# ---------------------------------------------------------------------------- #
#? Minimum Enclosing Circle

(x,y),radius = cv2.minEnclosingCircle(cnt)
center = (int(x),int(y))
radius = int(radius)
enclosing_circle = cv2.circle(image.copy(), center, radius, (0,0,255), 2)

show_image("Minimum Enclosing Circle", enclosing_circle)

# ---------------------------------------------------------------------------- #
#? Fitting an Ellipse

ellipse = cv2.fitEllipse(cnt)
fitted_ellipse = cv2.ellipse(image.copy(), ellipse, (0, 0, 255), 2)

show_image("Fitting an Ellipse", fitted_ellipse)

# ---------------------------------------------------------------------------- #
#? Fitting a Line

rows,cols = image.shape[:2]
[vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2,0,0.01,0.01)
lefty = int((-x*vy/vx) + y)
righty = int(((cols-x)*vy/vx)+y)
line_fitted = cv2.line(image.copy(), (cols-1,righty), (0,lefty), (0,0,255), 2)

show_image("Fitting a Line", line_fitted, destroy=True)

# ---------------------------------------------------------------------------- #