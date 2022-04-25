import cv2
import numpy as np
import matplotlib.pyplot as plt



class img_io:

    def read_img(img):
        return cv2.imread(img, 0)

    def show_img(img):
        fig = plt.figure(figsize=(12,10))
        ax = fig.add_subplot(111)
        ax.imshow(img, cmap='gray')
        plt.show()


target = img_io.read_img('/media/m2t/STORAGE/YEDEK/Belgeler/WS/Python/capstoneProject/Sidewalks/IMG_20220325_173544.jpg')
height_target, width_target = target.shape
resized_sidewalk = cv2.resize(target, (int(width_target / 7), int(height_target / 7)), interpolation= cv2.INTER_LINEAR)

pattern = img_io.read_img('/media/m2t/STORAGE/YEDEK/Belgeler/WS/Python/capstoneProject/Sidewalks/pattern.jpg')
height_pattern, width_pattern = pattern.shape
resized_pattern = cv2.resize(pattern, (int(width_pattern / 7), int(height_pattern / 7)), interpolation= cv2.INTER_LINEAR)


def featureMatching():
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(pattern,None)
        kp2, des2 = sift.detectAndCompute(target,None)

        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2, k=2)
        good = []

        for match1,match2 in matches:
            if match1.distance < 1*match2.distance:
                good.append([match1])

        # cv2.drawMatchesKnn expects list of lists as matches.
        sift_matches = cv2.drawMatchesKnn(pattern,kp1,target,kp2,good,None,flags=2)

        return sift_matches

img_io.show_img(featureMatching())