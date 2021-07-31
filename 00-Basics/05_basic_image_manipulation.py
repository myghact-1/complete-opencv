import cv2
import numpy as np

# --------------------------------- Todo List -------------------------------- #
#* 1. Color convertor
#* 2. Split image in color channels
#* 3. Image croping / Manipulating pixels / Accessing ROI
#* 4. Making Borders
# ---------------------------------------------------------------------------- #

image = cv2.imread('images\\2.jpg')
logo = cv2.imread("images\\OpenCV_logo.jpg")


def show_image(window, image, destroy=False):
    cv2.imshow(window, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------- #
#? color convertor

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
show_image("Original", image)
show_image("Gray", gray, destroy=True)

# ---------------------------------------------------------------------------- #
#? splitting image in color channels

(b, g, r) = cv2.split(logo)  # b = image[:,:,0]
show_image("Original Logo", logo)
show_image("Blue channel", b)
show_image("Green channel", g)
show_image("Red channel", r, destroy=True)

# ---------------------------------------------------------------------------- #
#? image croping and pixel manipulation

show_image("Original image", image)

# access the roi pixels / crop
roi = image[150:250, 200:300, :]
show_image("ROI", roi)

# manipulate pixels
image[150:250, 200:300, :] = 0
show_image("Manipulated image", image, destroy=True)

# ---------------------------------------------------------------------------- #
#? Making borders (Padding)

show_image("Original Logo", logo)

replecate = cv2.copyMakeBorder(logo, 10,10,10,10, cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(logo, 10,10,10,10, cv2.BORDER_REFLECT)
isolated = cv2.copyMakeBorder(logo, 10,10,10,10, cv2.BORDER_ISOLATED)
reflect101 = cv2.copyMakeBorder(logo, 10,10,10,10, cv2.BORDER_REFLECT101)
reflect_101 = cv2.copyMakeBorder(logo, 10,10,10,10, cv2.BORDER_REFLECT_101)
constant = cv2.copyMakeBorder(logo, 10,10,10,10, cv2.BORDER_CONSTANT, value=(255,255,0))
wrap = cv2.copyMakeBorder(logo, 10,10,10,10, cv2.BORDER_WRAP)

show_image("replecate", replecate)
show_image("reflect", reflect)
show_image("isolated", isolated)
show_image("reflect101", reflect101)
show_image("reflect_101", replecate)
show_image("constant", constant)
show_image("wrap", wrap, destroy=True)

# ---------------------------------------------------------------------------- #