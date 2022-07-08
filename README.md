A sidewalk tracker and obstacle detector.

Dependencies:

Numpy 1.20
Matplotlib 3.4.3
OpenCV 4.5.5.64
Tensorflow 2.9.0
Segmentation Models 1.0.1
pyttsx3 2.90
YOLO

Project initially created to help visually impaired people to navigate on sidewalks
But its quite modular and can be used in diferent tasks with minor tweaks

Must be used with a stereo camera system for depth mapping functionality
Optimized for Playstation Camera system
Calibration files can be found in cameraConfig folder


Other camera systems must be calibrated and left and right dimentions should defined accordingly

Capabilities

- Sidewalk detection and tracking functionalty uses one of the cameras in the stereo system
- Sidewalk detection made using binary semantic segmentation
- Segmentation Models library used for traning and detecting process
- Detection model uses Unet architecture with resnet34 backbone which trained with ImageNet
- Detection of 80 different objects with yolo model

There is also a module that gives feedbacks with speech using pyttsx3 library
Current build only contains Turkish phrases (can be found in messageTemplates)
New phrases in different languages can be created in json format

References:

sieuwe1's project helped a ton for depth functionality
https://github.com/sieuwe1/PS4-eye-camera-for-linux-with-python-and-OpenCV

Also
https://github.com/bigboss-ps3dev
https://github.com/ps4eye

Segmentation Models is a great library for semantic segmentation with transfer learning
https://github.com/qubvel/segmentation_models
