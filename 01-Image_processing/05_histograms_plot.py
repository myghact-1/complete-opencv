
# Image credits
# https://www.pexels.com/@arnie-watkins-1337313

import cv2
import numpy as np
import matplotlib.pyplot as plt

image  = cv2.imread("images\\desert.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# ---------------------------------------------------------------------------- #
#? Grayscale image histogram

hist = cv2.calcHist([gray], [0], None, [256], [0, 255])
fig = plt.figure(num=1, figsize=(10, 6))
plt.subplot(121)
plt.imshow(gray, cmap='gray')
plt.title("Grayscale image")
plt.subplot(122)
plt.plot(hist)
plt.title("Grayscale image histogram")
plt.show()

# ---------------------------------------------------------------------------- #
#? Color image Histogram

fig = plt.figure(num=2, figsize=(10,6))
colors = ['b','g','r']

for i, color in enumerate(colors):
    hist = cv2.calcHist([image], [i], None, [256], [0,256])
    plt.subplot(121)
    plt.plot(hist, color=color)
    
plt.subplot(122)
plt.imshow(image_rgb)
plt.show()

# ---------------------------------------------------------------------------- #
#! Exercise Mask Image