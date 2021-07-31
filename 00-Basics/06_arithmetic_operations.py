import cv2
import numpy as np

image = cv2.imread('images\\2.jpg')
logo = cv2.imread("images\\OpenCV_logo.jpg")

#! resize image same as logo image
image_resized = cv2.resize(image, (logo.shape[1], logo.shape[0]))

def show_image(window, image, destroy=False):
    cv2.imshow(window, image)
    cv2.waitKey(0)
    if destroy:
        cv2.destroyAllWindows()


# --------------------------- Arithmetic operations -------------------------- #

#? Arithmetic operations

x = np.uint8([250])
y = np.uint8([10])

print("Addition cv2", cv2.add(x,y) )    # 250+10 = 260 => 255
print("Addition normal", x+y )          # 250+10 = 260 % 256 = 4

print("Subraction cv2", cv2.subtract(y, x)) # 10-250 = -240 => 0
print("Subraction normal", y-x)             # 10-250 = 256 % 240 = 16

# ---------------------------- Bitwise operations ---------------------------- #
#? Bitwise operations

gray_image = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
gray_logo = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)


bitwise_not = cv2.bitwise_not(gray_logo)
bitwise_or = cv2.bitwise_or(gray_logo, gray_image)
bitwise_and = cv2.bitwise_and(gray_logo, gray_image)
bitwise_xor = cv2.bitwise_xor(gray_logo, gray_image)

show_image("Gray Logo", gray_logo)
show_image("Gray image", gray_image)
show_image('Bitwise-Not', bitwise_not)
show_image("Bitwise-OR", bitwise_or)
show_image("Bitwise-And", bitwise_and)
show_image("Bitwise-XOR", bitwise_xor, destroy=True)

# ------------------------------ Image Blending ------------------------------ #
#? image blending (same size images)

blended = cv2.addWeighted(logo, 0.7, image_resized, 0.3, 0)
show_image("Logo", logo)
show_image("Pizza", image_resized)
show_image("Blended Image", blended, destroy=True)