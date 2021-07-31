# https://docs.opencv.org/master/d5/d69/tutorial_py_non_local_means.html

#* Non-local Means Denoising algorithm
# ---------------------------------------------------------------------------- #
#* cv.fastNlMeansDenoising() - works with a single grayscale images
#* cv.fastNlMeansDenoisingColored() - works with a color image.
#* cv.fastNlMeansDenoisingMulti() - works with image sequence captured in short period of time (grayscale images)
#* cv.fastNlMeansDenoisingColoredMulti() - same as above, but for color images.
# ---------------------------------------------------------------------------- #

import cv2

image = cv2.imread("images\\desert.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dst1 = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
dst2 = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

# display all images
cv2.imshow("Original Image", image)
cv2.imshow("Gray image", gray)
cv2.imshow("Denoised color image", dst1)
cv2.imshow("Denoised Gray image", dst2)

cv2.waitKey(0)
cv2.destroyAllWindows()
