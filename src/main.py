from email import message
import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

# from depthMapping import createDepthMap
# from sidewalkDetection import trackSidewalks
from eventManager import getEvent
from voiceFeedback import speak
from objectRecognition import detect_image

# Image from prototype's camera contains left right section in one matrix
# here we divide it
def decode(frame):
    left = np.zeros((800,1264,3), np.uint8)
    right = np.zeros((800,1264,3), np.uint8)
    
    for i in range(800):
        left[i] = frame[i, 64: 1280 + 48] 
        right[i] = frame[i, 1280 + 48: 1280 + 48 + 1264] 
    
    return (left, right)


def main():
    #capture image from camera the "1" argument depends on the index of the video input device in the system
    # cap = cv2.VideoCapture('../DATA/sidewalk_samples/VID_20220315_153638.mp4')
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3448)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 808)

    while(True):

        # read image frame by frame in every iteration
        ret, frame = cap.read()

        # divide it as left and right
        left, right = decode(frame)

        # cv2.imshow('left',left)
        # cv2.imshow('right',right)
        # cv2.imshow('frame', frame)

        # createDepthMap function creates depth map from left and right images
        # creates an event if minimum distance threshold reached
        # depthMap = createDepthMap(left,right)

        # trackSidewalks takes one of the two image
        # creates an event acording to position
        # sidewalkMask = trackSidewalks(frame)

        detectObject = detect_image(left)
        
        # get the event with upmost priority
        user_msg = getEvent()
        if user_msg:
            speak(user_msg)

        
        # cv2.imshow('Live Depth Map', depthMap)

        # cv2.putText(sidewalkMask, "Status: " + str(user_msg),(5, 20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,0,255),2,cv2.LINE_AA)  
        # cv2.imshow('Sidewalk Mask', sidewalkMask)
        
        cv2.imshow('bounding_box', detectObject)
        

        # what x amount of time for the next frame
        time.sleep(1/30)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


main()