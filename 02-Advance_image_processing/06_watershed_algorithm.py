
# https://docs.opencv.org/master/d3/db4/tutorial_py_watershed.html

import cv2
import numpy as np

coins = cv2.imread("images\\coins.jpg")
gray = cv2.cvtColor(coins, cv2.COLOR_BGR2GRAY)

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()


# ---------------------------------------------------------------------------- #
#? Segmentation without Watershed Algorithm

blurred = cv2.medianBlur(gray, 5)

_, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)

# show_image("Original Image", coins)
# show_image("Threshold", thresh, destroy=True)

# find contours
contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

def draw_contours(contours, image):
    # draw contours
    for cnt in contours:
        cv2.drawContours(image, [cnt], 0, (255,0,0), 2)
        
    show_image("Total coins find", image, destroy=True)
    
# draw_contours(contours, coins.copy())


# ---------------------------------------------------------------------------- #
#? Segmentation with Watershed Algorithm
# --------------------------------- All Steps -------------------------------- #
#* 1. load the grayscale image
#* 2. convert grayscale image in binary image (thresholding)
#* 3. remove noise (morphological operations)
#* 4. find the sure foreground and sure background (distance transform)
#* 5. find the unknown region (subtract bg - fg)
#* 6. find all markers (connected component)
#* 7. markers = markers + 1
#* 8. watershed algirithm
# ---------------------------------------------------------------------------- #

_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


