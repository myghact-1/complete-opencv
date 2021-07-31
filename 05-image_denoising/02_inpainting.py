# https://docs.opencv.org/master/df/d3d/tutorial_py_inpainting.html


import cv2
from skimage.restoration import inpaint


image = cv2.imread("images\\woman_distorted.jpg")
mask_gray = cv2.imread("images\\woman_mask.jpg", 0)


result = cv2.inpaint(image, mask_gray, 10, cv2.INPAINT_TELEA)
result2 = cv2.inpaint(image, mask_gray, 10, cv2.INPAINT_NS)

# result = inpaint.inpaint_biharmonic(image, mask_gray, multichannel=True)

cv2.imshow("original image", image)
cv2.imshow("mask image", mask_gray)
cv2.imshow("Result with cv2.INPAINT_TELEA", result)
# cv2.imshow("Result with cv2.INPAINT_NS", result2)

cv2.waitKey(0)
cv2.destroyAllWindows()