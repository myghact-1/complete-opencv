
# Image credits
# https://www.pexels.com/@arnie-watkins-1337313

import cv2
import numpy as np
import matplotlib.pyplot as plt

image  = cv2.imread("images\\desert.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------- #
#? Histogram equalization

equalized = cv2.equalizeHist(gray)

fig = plt.figure(num=1, figsize=(10,6))
plt.subplot(121)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.subplot(122)
plt.imshow(equalized, cmap='gray')
plt.title("Equalized Image")
plt.show()


# ---------------------------------------------------------------------------- #
#? CLAHE (Contrast Limited Adaptive Histogram Equalization)
# adaptive histogram equalization

clahe = cv2.createCLAHE(clipLimit=2, tileGridSize=(8,8))
res = clahe.apply(gray)

fig = plt.figure(num=2, figsize=(10,6))
plt.subplot(121)
plt.imshow(gray, cmap='gray')
plt.title("Original Image")
plt.subplot(122)
plt.imshow(res, cmap='gray')
plt.title("CLAHE Equalized Image")
plt.show()

# ---------------------------------------------------------------------------- #
#! Histogram Backprojection