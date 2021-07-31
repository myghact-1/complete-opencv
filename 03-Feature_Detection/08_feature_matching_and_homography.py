# https://docs.opencv.org/master/d1/de0/tutorial_py_feature_homography.html

import cv2
import numpy as np

wine = cv2.imread("images\\wine.jpg")               # train image
wine_label = cv2.imread("images\\wine_label.jpg")   # query image

wine_gray = cv2.cvtColor(wine, cv2.COLOR_BGR2GRAY)
wine_label_gray = cv2.cvtColor(wine_label, cv2.COLOR_BGR2GRAY)

# initialize the orb
sift = cv2.SIFT_create()

# find the keypoints
kp1, des1 = sift.detectAndCompute(wine_gray, None)
kp2, des2 = sift.detectAndCompute(wine_label_gray, None)

FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks = 50)

flann = cv2.FlannBasedMatcher(index_params, search_params)

matches = flann.knnMatch(des2, des1, k=2)

# store all the good matches as per Lowe's ratio test.
good = []
for m,n in matches:
    if m.distance < 0.7*n.distance:
        good.append(m)

MIN_MATCH_COUNT = 10

if len(good) > MIN_MATCH_COUNT:
    src_pts = np.float32([ kp2[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
    dst_pts = np.float32([ kp1[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
    
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
    matchesMask = mask.ravel().tolist()
    
    h,w = wine_label_gray.shape
    
    pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
    dst = cv2.perspectiveTransform(pts,M)
    
    wine_gray = cv2.polylines(wine_gray,[np.int32(dst)],True,255,3, cv2.LINE_AA)
else:
    print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
    matchesMask = None
    
    
draw_params = dict(matchColor = (255,0,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)

result = cv2.drawMatches(wine_label_gray, kp2, wine_gray, kp1,
                         good, None, **draw_params)

cv2.imshow("features matched and homography", result)
cv2.waitKey(0)
cv2.destroyAllWindows()