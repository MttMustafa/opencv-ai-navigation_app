import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import os

from eventManager import addEvent

#Load camera parameters
ret = np.load('../cameraConfig/param_ret.npy')
K = np.load('../cameraConfig/param_K.npy')
dist = np.load('../cameraConfig/param_dist.npy')
h,w = (800, 1264)
new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(K,dist,(w,h),1,(w,h))

mapx, mapy = cv2.initUndistortRectifyMap(K,dist, None ,new_camera_matrix,(w, h),cv2.CV_16SC2)

kernel= np.ones((13,13),np.uint8)

#Stereo matcher settings
win_size = 5
min_disp = 10
max_disp = 16 * 2 + 10
num_disp = max_disp - min_disp # Needs to be divisible by 16
stereo = cv2.StereoSGBM_create(minDisparity= min_disp,
numDisparities = num_disp,
blockSize = 5,
uniquenessRatio = 10,
speckleWindowSize = 1000,
speckleRange = 10,
disp12MaxDiff = 25,
P1 = 8*3*win_size**2,#8*3*win_size**2,
P2 =32*3*win_size**2) #32*3*win_size**2)


def createDepthMap(left, right):

    #downsample image for higher speed
    img_1_downsampled = cv2.pyrDown(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY))
    img_2_downsampled = cv2.pyrDown(cv2.cvtColor(right, cv2.COLOR_BGR2GRAY))

    new_w, new_h = img_1_downsampled.shape

    #compute stereo
    disp = stereo.compute(img_1_downsampled,img_2_downsampled)
    
    #denoise step 1
    denoised = ((disp.astype(np.float32)/ 16)-min_disp)/num_disp
    dispc= (denoised-denoised.min())*255
    dispC= dispc.astype(np.uint8)     
    
    #denoise step 2
    denoised= cv2.morphologyEx(dispC,cv2.MORPH_CLOSE, kernel)
    
    #apply color map
    disp_Color= cv2.applyColorMap(denoised,cv2.COLORMAP_OCEAN)
    
    f = 0.3*w   # Focal length
    Q = np.float32([[1, 0, 0, -0.5*new_w],
                    [0,-1, 0,  0.5*new_h], # turn points 180 deg around x-axis,
                    [0, 0, 0,      f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv2.reprojectImageTo3D(disp, Q)

    z_values = points[:,216:416,2]
    z_values = z_values.flatten()
    indices = z_values.argsort()

    min_distance = np.mean(np.take(z_values,indices[0:25280]))                             

    # return message on close encounters
    obstacle_status = False
    if min_distance <= 0.7:
        obstacle_status = True
    else:
        obstacle_status = False
    
    if obstacle_status:
        addEvent('od', time.time())

    return disp_Color