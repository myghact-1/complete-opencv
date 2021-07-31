import cv2
from skimage import exposure

src_image = cv2.imread("images\\cow.jpg") # image to change
# https://www.pexels.com/@abdghat
ref_image = cv2.imread("images\\sunset.jpg") # color to be applied

multi = True if src_image.shape[-1] >1 else False
matched = exposure.match_histograms(src_image, ref_image, multichannel=multi)

cv2.imshow("Source Image", src_image)
cv2.imshow("Reference Image", ref_image)
cv2.imshow("Matched image", matched)

cv2.waitKey(0)
cv2.destroyAllWindows()