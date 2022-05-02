import cv2
import numpy as np
import matplotlib.pyplot as plt
from soupsieve import match
import sys
import math
import time
from sqlalchemy import null
from sympy import laplace_transform

# from imageInputOutput import img_io
# from imageProcessers import img_proc
from featureMatchers import feature_matchers

class img_io:

    def read_img(img, resize = True):
    
        img = cv2.imread(img, 0)
        if img is None:
            print("Image not found")
            sys.exit()
        if resize:
            return img_io.resize_img(img)
        else: return img

    def resize_img(img):
        ratio = 7
        img = cv2.resize(img, (int(img.shape[1] / ratio), int(img.shape[0] / ratio)), interpolation= cv2.INTER_CUBIC)
        return img
            
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
            frame = img_io.resize_img(frame)

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