import os

import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from config import MIN_MATCH_COUNT

# Initiate SIFT detector
sift = cv.xfeatures2d.SIFT_create()
FLANN_INDEX_KDTREE = 0
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)

flann = cv.FlannBasedMatcher(index_params, search_params)


def ensure_folder(folder):
    if not os.path.isdir(folder):
        os.mkdir(folder)


def do_match(file1, file2):
    img1 = cv.imread(file1, 0)
    img2 = cv.imread(file2, 0)

    print('img1.shape: ' + str(img1.shape))
    print('img2.shape: ' + str(img1.shape))

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    print('len(good): ' + str(len(good)))

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        H, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        print(H)

        draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                           singlePointColor=None,
                           matchesMask=mask,  # draw only inliers
                           flags=2)

        img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)

        plt.imshow(img3, 'gray'), \
        plt.show()

    return None
