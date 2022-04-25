import cv2
import numpy as np
import matplotlib.pyplot as plt
from soupsieve import match
import sys
import math
import time

from sqlalchemy import true
from sympy import laplace_transform

class img_io:

    def read_img(img, resize = True):
        if type(img) == str:
            img = cv2.imread(img, 0)
        if resize:
            return img_proc.resize_img(img)
        else: return img
        
            
    def show_img(img):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        plt.show()

    def play_vid(pattern, vid):
        
        capture = cv2.VideoCapture(vid)

        if capture.isOpened() == False:
            print('ERROR: VIDEO FILE NOT FOUND OR WRONG CODEC!')

        while capture.isOpened() == True:

            ret, frame = capture.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # frame = img_proc.auto_canny(frame)
            frame = img_proc.resize_img(frame)

            if ret == True:
                fps = 30
                time.sleep(1 / fps) 

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
                frame = feature_matchers.flann_match(pattern, frame)

                cv2.imshow('frame', frame)

            else: break
                
        capture.release()
        cv2.destroyAllWindows()

class img_proc:

    def resize_img(img):
        ratio = 7
        img = cv2.resize(img, (int(img.shape[1] / ratio), int(img.shape[0] / ratio)), interpolation= cv2.INTER_LINEAR)
        return img

    def blur_smooth(img): 
        blurred_img = cv2.GaussianBlur(img_io.read_img(img), (7, 7), 0)
        return blurred_img
        
    def manual_canny(img):
        if type(img) == str:
            img = img_proc.blur_smooth(img)
        ath = cv2.adaptiveThreshold(img.copy(), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 101, 35)
        return ath

    def sobel_op(img):
        sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize = 3)
        # sobely = cv2.Sobel(img_proc.blur_smooth(img), cv2.CV_64F, 0, 1, ksize= 9)
        # blended_1 = cv2.addWeighted(src1 = sobelx.copy(), alpha = 0.5, src2 = sobely.copy(), beta = 0.5, gamma = 0)
        return sobelx

    def auto_canny(img):
        median_val = np.median(img_io.read_img(img))
        lower_t = int(max(0, 5 * median_val)) #lower threshold is either 0 or 70% of median whichever is bigger
        upper_t = int(min(255, 21* median_val)) #upper threshold is either 255 or 130% whichever is smaller
        edges = cv2.Canny(image = img_io.read_img(img), threshold1 = lower_t, threshold2 = upper_t + 45)
        return edges

    def laplacian(img):
        laplacian = cv2.Laplacian(img_proc.blur_smooth(img), cv2.CV_64F)
        return laplacian

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

def main():
    
    target = img_io.read_img('/media/m2t/STORAGE/YEDEK/Belgeler/WS/Python/capstoneProject/Sidewalks/sidewalk/IMG_20220325_173537.jpg')     

    laplacian = img_proc.sobel_op(target)
    
    abs_grad = cv2.convertScaleAbs(laplacian)
    # grad_norm = (laplacian * 10 / laplacian.max()).astype(np.uint8)

    mct = img_proc.manual_canny(abs_grad)
    
    # lines = cv2.HoughLinesP(mct, 1, np.pi / 180, 1, minLineLength=50, maxLineGap=5)

    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(mct, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    img_io.show_img(mct)


    # pattern = img_io.read_img('/media/m2t/STORAGE/YEDEK/Belgeler/WS/Python/capstoneProject/Sidewalks/walkPattern/sw.png', True)
    # img_io.play_vid(pattern, '/media/m2t/STORAGE/YEDEK/Belgeler/WS/Python/capstoneProject/Sidewalks/VID_20220315_153638.mp4')

main()