
# https://docs.opencv.org/master/d6/d10/tutorial_py_houghlines.html


import cv2
import numpy as np

image = cv2.imread("images\\sudoku.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Hough Line Transform

# Change image to binary (threshold) or just use the canny edge detection

edges = cv2.adaptiveThreshold(gray, 255, 
                              cv2.ADAPTIVE_THRESH_MEAN_C, 
                              cv2.THRESH_BINARY_INV, 5, 5)

# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
show_image("Edges", edges)

# detect the lines
lines = cv2.HoughLines(edges, 1, np.pi/180, 200)
# draw all lines
for line in lines:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho
    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))
    
    cv2.line(image, (x1,y1), (x2,y2), (0,0,255), 2)
    
show_image("Hough Line Transformed Image", image)

# ---------------------------------------------------------------------------- #
#? Probabilistic Hough Transform
# Probabilistic Hough Transform is an optimization of the Hough Transform

# Change image to binary (threshold) or just use the canny edge detection

edges = cv2.adaptiveThreshold(gray, 255, 
                              cv2.ADAPTIVE_THRESH_MEAN_C, 
                              cv2.THRESH_BINARY_INV, 5, 5)

# edges = cv2.Canny(gray, 50, 150, apertureSize=3)
# show_image("Edges", edges)

# detect the lines
lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                        minLineLength=100,
                        maxLineGap=10)

# draw all lines
for line in lines:
    x1,y1,x2,y2 = line[0]
    
    cv2.line(image, (x1,y1), (x2,y2), (0,255,0), 2)
    
show_image("Probabilistic Hough Transform", image)