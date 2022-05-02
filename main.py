import cv2
import numpy as np
import matplotlib.pyplot as plt
from soupsieve import match
import sys
import math
import time

from sqlalchemy import true
from sympy import laplace_transform

from imageInputOutput import img_io
from imageProcessers import img_proc
from featureMatchers import feature_matchers

def main():
    
    target = img_io.read_img('Python/capstoneProject/Sidewalks/sidewalk/IMG_20220325_173537.jpg')     

    laplacian = img_proc.sobel_op(target)
    
    abs_grad = cv2.convertScaleAbs(laplacian)
    # grad_norm = (laplacian * 10 / laplacian.max()).astype(np.uint8)

    mct = img_proc.manual_canny(abs_grad)
    
    # lines = cv2.HoughLinesP(mct, 1, np.pi / 180, 1, minLineLength=50, maxLineGap=5)

    # for line in lines:
    #     x1, y1, x2, y2 = line[0]
    #     cv2.line(mct, (x1, y1), (x2, y2), (255, 0, 0), 3)
    
    # img_io.show_img(mct)


    pattern = img_io.read_img('Python/capstoneProject/Sidewalks/walkPattern/pattern.jpg', True)
    img_io.play_vid(pattern, 'Python/capstoneProject/Sidewalks/VID_20220315_153638.mp4')

main()