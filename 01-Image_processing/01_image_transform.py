import cv2
import imutils
import numpy as np

# --------------------------------- All Tasks -------------------------------- #
#* 1. Image resizing, Image Pyramids
#* 2. Image flipping
#* 3. Image Rotation
#* 4. Image Translation (Affine & Perspective)
# ---------------------------------------------------------------------------- #

#? image reading
image = cv2.imread("images\\2.jpg")

def show_image(winname, image, destroy=False):
    cv2.imshow(winname, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()

# ---------------------------------------------------------------------------- #
#? image resizing (Pyramids)

resized_image = cv2.resize(image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
pyr_up = cv2.pyrUp(image)
pyr_down = cv2.pyrDown(image)

show_image("Original Image", image)
show_image("Resized image (half)", resized_image)
show_image("Pyramids up image", pyr_up)
show_image("Pyramids down image", pyr_down, destroy=True)


# ---------------------------------------------------------------------------- #
#? image flipping

flipped = cv2.flip(image, 1) # Horizontally flipped image
show_image("Original Image", image)
show_image('Flipped image', flipped, destroy=True)

# ---------------------------------------------------------------------------- #
#? rotate image

rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
show_image("Original Image", image)
show_image('90 degree Rotated image', rotated)

# get a specific rotation matrix
M = cv2.getRotationMatrix2D((50,50), -45, 1)
rotated_2 = cv2.warpAffine(image, M, (500,600))
show_image('-45 degree Rotated image', rotated_2)

# complete image bound result
rotated_3 = imutils.rotate_bound(image, -45)
show_image('-45 degree Rotated bound image', rotated_3, destroy=True)

# ---------------------------------------------------------------------------- #
#? Image translation

#! Translation
M = np.float32([[1, 0, 100],
                [0, 1, 60]])

(H, W, C) = image.shape
translated = cv2.warpAffine(image, M, (W, H))

show_image("Original Image", image)
show_image("Translated Image", translated)

#! Affine Matrix
pt1 = np.float32([[50,50], [150, 50], [50, 150]])
pt2 = np.float32([[20,70], [100, 30], [100, 100]])

M = cv2.getAffineTransform(pt1, pt2)
affined = cv2.warpAffine(image, M, (W, H))
show_image("Affined Image", affined)

#! Perpective Matrix
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])

M = cv2.getPerspectiveTransform(pts1, pts2)

perspective = cv2.warpPerspective(image, M, (300,300))
show_image("Perspective image", perspective, destroy=True)