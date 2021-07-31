
# https://docs.opencv.org/master/da/d22/tutorial_py_canny.html

import cv2
import numpy as np

image  = cv2.imread("images\\opencv1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Photo by Julia Volk from Pexels -- https://www.pexels.com/@julia-volk
image2 = cv2.imread("images\\cow.jpg")
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Canny Edge Detection

blurred = cv2.GaussianBlur(gray, (5,5), 0)
edges = cv2.Canny(blurred, 100, 180)

blurred2 = cv2.GaussianBlur(gray2, (3,3), 0)
edges2 = cv2.Canny(blurred2, 127, 180)


show_image("Original Image 2", gray)
show_image("Edges 1", edges)
show_image("Original Image 2", gray2)
show_image("Edges", edges2, destroy=True)

# ---------------------------------------------------------------------------- #
#? Create Trackbar for edge detection

# The function attached to the trackbar.
def func(pos):
    print(pos)

cv2.namedWindow("edges")
cv2.createTrackbar('TH1', "edges", 0, 255, func)
cv2.createTrackbar('TH2', "edges", 0, 255, func)

while True:

    p1 = cv2.getTrackbarPos("TH1","edges")
    p2 = cv2.getTrackbarPos("TH2","edges")
    
    edges = cv2.Canny(blurred2, p1, p2)
    
    cv2.imshow("edges", edges)
    
    if cv2.waitKey(1)==ord('q'):
        break
    
cv2.destroyAllWindows()