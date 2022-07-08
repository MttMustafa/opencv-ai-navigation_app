import matplotlib.pyplot as plt
import segmentation_models as sm
import numpy as np
import albumentations as A
import cv2
import sys
import time

from eventManager import addEvent

BACKBONE = 'resnet34'
CLASSES = ['pavement']

preprocess_input = sm.get_preprocessing(BACKBONE)
model = sm.Unet(BACKBONE, classes=1, activation='sigmoid')
model.load_weights('../models/pavement_modelv3_resnet34.h5')


def navigate(pr_mask):

    message = []

    mask_dimen = pr_mask.shape
    mask_dimen_y = mask_dimen[1] #rows
    mask_dimen_x = mask_dimen[2] #columns
    horizontal_center = int(mask_dimen_x / 2)
    image_array = pr_mask[0]

    # Clearing black bar often drawn in the right most column of the image
    for i in range(0, mask_dimen_y):
        image_array[i][mask_dimen_x - 1 :mask_dimen_x] = 1


    #Finding black pixel's average x coordinate
    cluster_sum = 0
    cluster_pixel_count = 0
    for i in range(0, mask_dimen_y):
        for j in range(0, mask_dimen_x):
            if image_array[i][j][0] == 0:
                cluster_sum = cluster_sum + j
                cluster_pixel_count = cluster_pixel_count + 1
    cluster_horizontal_center = int(cluster_sum / cluster_pixel_count)

    # TRACK SIDEWALK

    # take first 25% of the vertical dimention
    # iterate from bottom to top
    # if there was no black pixels there is not sidewalk

    # if there was black pixels track rest of the dimention
    # if white pixel detected check right and left to find black pixels
        # if not found sidewalk ends there
        # if pixel was found in one side it means sidewalk curves to that direction

    is_sidewalk_detected = False
    count_from_base = 0
    base_portion = int(mask_dimen_y * 25 / 100)

    for i in range(mask_dimen_y - 1, (mask_dimen_y - base_portion), -1):
        if image_array[i][horizontal_center][0] == 0:
            count_from_base = count_from_base + 1

    if count_from_base > base_portion / 2:
        is_sidewalk_detected = True
    else: is_sidewalk_detected = False

    # message.append(('sw-on', time.time()))

    found_rest = False
    if is_sidewalk_detected:
        for i in range(base_portion + 1, 0, -1):
            if(image_array[i][horizontal_center][0] == 1):
                message.append(('sw-end', time.time()))
                for j in range(horizontal_center + 1, mask_dimen_x):
                    if(image_array[i][j][0] == 0) and (cluster_horizontal_center > horizontal_center):
                        message.append(('sw-rt', time.time()))
                        found_rest = True
                        break
                if found_rest == False:
                    for j in range(horizontal_center - 1, 0, -1):
                        if(image_array[i][j][0] == 0) and (cluster_horizontal_center < horizontal_center):
                            message.append(('sw-lt', time.time()))
                            found_rest = True
                            break
                break
    return message


def trackSidewalks(frame):

    #resize frame to a smaller scale (also must be divedable to 2)
    frame = cv2.resize(frame, (512, 512), interpolation= cv2.INTER_CUBIC)
    #package it into a list
    frame = np.expand_dims(frame, axis=0)
    # use segmentation model to create sidewalk mask
    pr_mask = model.predict(frame).round()
    frame = pr_mask
    # call navigate to get sidewalk state
    messages = navigate(frame)

    # navigate returns event time and and event id
    # add it to event list
    if messages:
        for msg in messages:
            addEvent(msg[0], msg[1])

    return frame[0]
  

                    