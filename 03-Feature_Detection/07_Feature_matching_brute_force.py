# https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html

import cv2
# import numpy as np

wine = cv2.imread("images\\wine.jpg")               # train image
wine_label = cv2.imread("images\\wine_label.jpg")   # query image

wine_gray = cv2.cvtColor(wine, cv2.COLOR_BGR2GRAY)
wine_label_gray = cv2.cvtColor(wine_label, cv2.COLOR_BGR2GRAY)

# initialize the orb
orb = cv2.ORB_create()

# find the keypoints
kp1, des1 = orb.detectAndCompute(wine_gray, None)
kp2, des2 = orb.detectAndCompute(wine_label_gray, None)

# create the brute force matcher
bf = cv2.BFMatcher_create(normType=cv2.NORM_HAMMING2, crossCheck=True)

# match the descriptors
matches = bf.match(des2, des1)

# sort the matches by distance
matches = sorted(matches, key= lambda x: x.distance)

# draw the top 20 matches
result = cv2.drawMatches(wine_label_gray, kp2, wine_gray, kp1, matches[:20], None,
                         matchColor=(0,0,255),
                         flags=cv2.DrawMatchesFlags_DEFAULT) # use the different flags


cv2.imshow("Feature matched image", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------------------------------------------------------------- #
#? Brute-Force Matching with SIFT Descriptors and Ratio Test

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(wine_gray, None)
kp2, des2 = sift.detectAndCompute(wine_label_gray, None)

# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des2, des1, k=2)

# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])
        
# draw the top 20 matches

result2 = cv2.drawMatchesKnn(wine_label_gray, kp2, wine_gray, kp1, good, None,
                             matchColor=(255,0,0),
                             flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

cv2.imshow("SIFT feature des. matched", result2)
cv2.waitKey(0)
# cv2.destroyAllWindows()

# ---------------------------------------------------------------------------- #
#? FLANN based Matcher

# Initiate SIFT detector
sift = cv2.SIFT_create()

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(wine_gray, None)
kp2, des2 = sift.detectAndCompute(wine_label_gray, None)

# FLANN parameters
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
search_params = dict(checks=50)   # or pass empty dictionary
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(des2, des1, k=2)

# Need to draw only good matches, so create a mask
matchesMask = [[0,0] for i in range(len(matches))]

# ratio test as per Lowe's paper
for i,(m,n) in enumerate(matches):
    if m.distance < 0.7*n.distance:
        matchesMask[i]=[1,0]

draw_params = dict(matchColor = (255,0,0),
                   singlePointColor = (255,0,0),
                   matchesMask = matchesMask,
                   flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)


result3 = cv2.drawMatchesKnn(wine_label_gray, kp2,wine_gray, kp1, 
                             matches, None, **draw_params)

cv2.imshow("FLANN based Matcher", result3)
cv2.waitKey(0)
cv2.destroyAllWindows()