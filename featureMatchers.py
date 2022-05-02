import cv2
import numpy as np
import matplotlib.pyplot as plt
from soupsieve import match
import sys
import math
import time

from sqlalchemy import true
from sympy import laplace_transform

class feature_matchers:

    def sift_match(pattern, sidewalk):

        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(pattern,None)
        kp2, des2 = sift.detectAndCompute(sidewalk,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        good = []

        for match1,match2 in matches:
            if match1.distance < 0.75*match2.distance:
                good.append([match1])

        # cv2.drawMatchesKnn expects list of lists as matches.
        sift_matches = cv2.drawMatchesKnn(pattern,kp1,sidewalk,kp2,good,None,flags=2)

        return sift_matches

    def orb_match(pattern, sidewalk):
        # Initiate ORB detector
        orb = cv2.ORB_create()

        # find the keypoints and descriptors with ORB
        kp1, des1 = orb.detectAndCompute(pattern,None)
        kp2, des2 = orb.detectAndCompute(sidewalk,None)

        # create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Match descriptors.
        matches = bf.match(des1,des2)

        # Sort them in the order of their distance.
        # matches = sorted(matches, key = lambda x:x.distance)

        # Draw first 25 matches.
        orb_matches = cv2.drawMatches(pattern,kp1,sidewalk,kp2,matches[:25],None,flags=2)

        return orb_matches

    def flann_match(pattern, sidewalk):

        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(pattern,None)
        kp2, des2 = sift.detectAndCompute(sidewalk,None)

        # FLANN parameters
        FLANN_INDEX_KDTREE = 0
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks=50)  

        flann = cv2.FlannBasedMatcher(index_params,search_params)

        matches = flann.knnMatch(des1,des2,k=2)

        good = []

        # ratio test
        for i,(match1,match2) in enumerate(matches):
            if match1.distance < .8*match2.distance:
                
                good.append([match1])

        # print(len(good))
        if len(good) >= 30:
            print('High number of pattern detection. Potential sidewalk!')

        flann_matches = cv2.drawMatchesKnn(pattern,kp1,sidewalk,kp2,good,None,flags=0)

        return flann_matches
