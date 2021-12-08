import numpy as np
import cv2

def reorder_points(points):
    points = points.squeeze()
    rectangle = np.zeros(shape=(4,2), dtype=np.float32)
    s = points.sum(axis=1)
    
    rectangle[0] = points[s.argmin()]
    rectangle[3] = points[s.argmax()]
    
    d = np.diff(points, axis=1)
    rectangle[1] = points[d.argmin()]
    rectangle[2] = points[d.argmax()]
    
    return rectangle

def get_selected_roi(image, points):
    
    rect = reorder_points(points)
    tl, tr, bl, br = rect
    
    w1 = tr[0] - tl[0]
    w2 = br[0]-bl[0]
    width = int(max([w1, w2]))
    
    h1 = bl[1]-tl[1]
    h2 = br[1]-tr[1]
    height = int(max([h1, h2]))
    
    pts1 = np.float32(rect)
    pts2 = np.float32([[0,0], [width, 0], [0, height], [width, height]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    dst = cv2.warpPerspective(image, M, (width, height))
    return dst