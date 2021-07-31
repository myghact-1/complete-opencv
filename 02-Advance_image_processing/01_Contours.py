
# https://docs.opencv.org/master/d3/d05/tutorial_py_table_of_contents_contours.html

import cv2
import numpy as np

image  = cv2.imread("images\\shapes.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        
# ---------------------------------------------------------------------------- #
#? Finding and Drawing Countors

# converting grayscale image in binary (black background, white foreground)
_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
# find all the contours in threshold image
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

print("Total contours found: ", len(contours))

# draw all countours

result_all = cv2.drawContours(image.copy(), contours, -1, (0,255,0), 3) # -1 to draw all contours
result_4 = cv2.drawContours(image.copy(), contours, 3, (0,255,0), 3) # drawing 4th contour

# Or
cnt = contours[2]
result_3 = cv2.drawContours(image.copy(), [cnt], 0, (0,255,0), 3) # drawing 3rd contours

# show_image('Threshold', thresh)
# show_image("All contours", result_all)
# show_image("Forth contour", result_4)
# show_image("Third contour", result_3, destroy=True)

# ---------------------------------------------------------------------------- #
#? Contour features

#* 1. Center of each contour

def draw_centers(contours):
    for i in range(len(contours)):
        cnt = contours[i]
        # Moments
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        centers = cv2.circle(image, (cx, cy), 5, (0,255,0), -1)
        show_image("Centers", centers)

# draw_centers(contours)

# ---------------------------------------------------------------------------- #

#* 2. Area and Perimeters
# sort contours by area descending

contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)

def draw_contours_area(contours):
    
    for i in range(len((contours))):
        cnt = contours[i]
        # area
        area = cv2.contourArea(cnt)
        result = cv2.drawContours(image, [cnt], 0, (0, 255, 0), 3)
        # Moments
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center = (cx, cy)
        cv2.putText(result, f"{area:0g}", center, cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 2)
        show_image("Countors by Area", result)
        
# draw_contours_area(contours)

# ---------------------------------------------------------------------------- #

contours_perimeter = sorted(contours, key=lambda x: cv2.arcLength(x, True), reverse=True)

def draw_contours_perimeter(contours):
    
    for i in range(len((contours))):
        cnt = contours[i]
        result = cv2.drawContours(image, [cnt], 0, (0, 255, 0), 3)
        # Moments
        M = cv2.moments(cnt)
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        center = (cx, cy)
        arc = cv2.arcLength(cnt, True)
        cv2.putText(result, f"{i+1}-{arc:.1f}", center, cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        show_image("Countors by Area", result)
        
# draw_contours_perimeter(contours_perimeter)

# ---------------------------------------------------------------------------- #
