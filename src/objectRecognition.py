import os
import time
import cv2
import numpy as np
from yolo_model import YOLO
from eventManager import addEvent

yolo = YOLO(0.6, 0.5)
file = '../DATA/coco_classes.txt'

def process_image(img):
    
    image = cv2.resize(img, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)

    return image

def get_classes(file):

    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names

all_classes = get_classes(file)

def draw(image, boxes, scores, classes, all_classes):
 
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))

        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        # print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        # print('box coordinate x,y,w,h: {0}'.format(box))
        return all_classes[cl]


def detect_image(image):

    image = process_image(image)

    start = time.time()
    boxes, classes, scores = yolo.predict(image, image.shape)
    end = time.time()

    # print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:
        object = draw(image, boxes, scores, classes, all_classes)
        addEvent(object,end)


    return image[0]