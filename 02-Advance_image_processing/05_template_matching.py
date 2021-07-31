
# https://docs.opencv.org/master/d4/dc6/tutorial_py_template_matching.html

import cv2
import numpy as np

image = cv2.imread('images\\desert.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
template = cv2.imread('images\\desert_template.jpg')
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

(h, w) = template.shape[:2]

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()
        

show_image("Original Image", image)
show_image("Template Image", template, destroy=True)
        
# ---------------------------------------------------------------------------- #
#? Template matching by using all different-2 methods

methods = ['cv2.TM_CCOEFF', 'cv2.TM_CCOEFF_NORMED', 'cv2.TM_CCORR',
           'cv2.TM_CCORR_NORMED', 'cv2.TM_SQDIFF', 'cv2.TM_SQDIFF_NORMED']

#! If you are using cv2.TM_SQDIFF as comparison method, minimum value gives the best match.

for meth in methods:
    method = eval(meth)
    
    # apply template matching
    matched = cv2.matchTemplate(gray, gray_template, method)
    # find the min-max values and locations
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(matched)
    
    # 
    if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
        top_left = min_loc
    else:
        top_left = max_loc
        
    bottom_right = (top_left[0]+w, top_left[1]+h)
    
    # draw rectangle
    result = cv2.rectangle(image.copy(), top_left, bottom_right, (255,0,0), 2)
    
    show_image(f"Matched template by method - {meth}", result)
    
# ---------------------------------------------------------------------------- #
#! Template Matching with Multiple Objects
