# https://docs.opencv.org/master/da/d53/tutorial_py_houghcircles.html


import cv2
import numpy as np

image = cv2.imread("images\\OpenCV_logo.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Hough Circle Transform

thresh = cv2.adaptiveThreshold(gray, 255,
                               cv2.ADAPTIVE_THRESH_MEAN_C,
                               cv2.THRESH_BINARY_INV,
                               5, 5)

# _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

# show_image("Threshold", thresh)

blurred = cv2.GaussianBlur(thresh, (5,5), 0)


circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, 1, 2,
                           50, 30)

circles = np.uint16(np.around(circles)) # three dimension
print('circles: ', circles)

# draw all circles
for i in circles[0, :]:
    # draw outer circle
    cv2.circle(image, (i[0], i[1]), i[2], (0,0,0), 3)
    # draw center
    cv2.circle(image, (i[0], i[1]), 3, (0,0,0), 3)
    
show_image("Circles Detected", image, destroy=True)