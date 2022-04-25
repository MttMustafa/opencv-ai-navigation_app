import cv2
import time
import numpy as np

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
        if match1.distance < .85*match2.distance:
            
            good.append([match1])

    flann_matches = cv2.circle(sidewalk, kp2, 5, (255,0,0), -1)
    # flann_matches = cv2.drawMatchesKnn(pattern,kp1,sidewalk,kp2,good,None,flags=0)

    return flann_matches

capture = cv2.VideoCapture('/media/m2t/STORAGE/YEDEK/Belgeler/WS/Python/capstoneProject/Sidewalks/VID_20220315_153638.mp4')

# capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if capture.isOpened() == False:
    print('ERROR: VIDEO FILE NOT FOUND OR WRONG CODEC!')

ret, frame1 = capture.read()

# prevImg = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

# hsv_mask = np.zeros_like(frame1)

# hsv_mask[:, :, 1] = 255

while capture.isOpened() == True:
    ret, frame2 = capture.read()
    
    if ret == True:
        fps = 30 #target videos vps value
        time.sleep(1 / 30) #adds 1/30 seconds delay before each frame so video can be played real time
        
        # nextImg = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        # flow = cv2.calcOpticalFlowFarneback(prevImg, nextImg, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        # mag, ang = cv2.cartToPolar(flow[:, :, 0], flow[:, :, 1], angleInDegrees = True)
        # hsv_mask[:, :, 0] = ang / 2
        # hsv_mask[:, :, 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        # bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_HSV2BGR)
    

        cv2.imshow('frame', frame2)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        # prevImg = nextImg
    else:
        break
        
capture.release()
cv2.destroyAllWindows()