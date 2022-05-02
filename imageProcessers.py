import cv2
import numpy as np
import matplotlib.pyplot as plt
from soupsieve import match
import sys
import math
import time
from sympy import laplace_transform


class img_proc:
    
    def blur_smooth(img):

        if type(img) is not np.ndarray:
            print("Blurring failed: Wrong format -> {dataType}".format(dataType = type(img)))
            sys.exit()
        
        blurred_img = cv2.GaussianBlur(img, (7, 7), 0)
        return blurred_img

    def manual_canny(img):

        if type(img) is not np.ndarray:
            print("Manual canny failed: Wrong format -> {dataType}".format(dataType = type(img)))
            sys.exit()

        img = img_proc.blur_smooth(img)
        ath = cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 35)
        return ath

    def sobel_op(img):

        if type(img) is not np.ndarray:
            print("Sobel operation failed: Wrong format -> {dataType}".format(dataType = type(img)))
            sys.exit()

        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
        # sobely = cv2.Sobel(img_proc.blur_smooth(img), cv2.CV_64F, 0, 1, ksize= 9)
        # blended_1 = cv2.addWeighted(src1 = sobelx.copy(), alpha = 0.5, src2 = sobely.copy(), beta = 0.5, gamma = 0)
        return sobelx

    def auto_canny(img):

        if type(img) is not np.ndarray:
            print("Auto canny failed: Wrong format -> {dataType}".format(dataType = type(img)))
            sys.exit()

        median_val = np.median(img)
        lower_t = int(max(0, 5 * median_val)) #lower threshold is either 0 or 70% of median whichever is bigger
        upper_t = int(min(255, 21* median_val)) #upper threshold is either 255 or 130% whichever is smaller
        edges = cv2.Canny(img, threshold1 = lower_t, threshold2 = upper_t + 45)
        return edges

    def laplacian(img):

        if type(img) is not np.ndarray:
            print("Laplacian operator failed: Wrong format -> {dataType}".format(dataType = type(img)))
            sys.exit()

        laplacian = cv2.Laplacian(img_proc.blur_smooth(img), cv2.CV_64F)
        return laplacian
