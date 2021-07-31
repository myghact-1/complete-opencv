import cv2

image = cv2.imread("images\\shapes.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

all_colormap_methods = [cmap for cmap in dir(cv2) if cmap.startswith('COLORMAP')]

print(all_colormap_methods)

for cmap in all_colormap_methods:
    method_name = 'cv2.'+cmap
    method = eval(method_name)
    
    result = cv2.applyColorMap(gray.copy(), colormap=method)
    
    cv2.imshow(f"image with {method_name}", result)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()