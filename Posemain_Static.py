import math
import cv2
import numpy as np
from time import time
import mediapipe as mp
import matplotlib.pyplot as plt

mp_pose = mp.solutions.pose

# Setting up Pose function.
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.3, model_complexity=2)

mp_drawing = mp.solutions.drawing_utils 

def detectPose(image, pose, display=True):
    output_image = image.copy()
    
    #BGR into RGB format.
    imageRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    #Pose Detection.
    results = pose.process(imageRGB)
    
    #height and width of the input image.
    height, width, _ = image.shape
    
    landmarks = []
    
    if results.pose_landmarks:
    
        # Draw Pose landmarks on the output image.
        mp_drawing.draw_landmarks(image=output_image, landmark_list=results.pose_landmarks,connections=mp_pose.POSE_CONNECTIONS)
        
        # Iterate over the detected landmarks.
        for landmark in results.pose_landmarks.landmark:            
            landmarks.append((int(landmark.x * width), int(landmark.y * height),int(landmark.z * width)))
    
    if display:
    
        # Display the original input image and the resultant image.
        plt.figure(figsize=[22,22])
        plt.subplot(121);plt.imshow(image[:,:,::-1]);plt.title("Original Image");plt.axis('off');
        plt.subplot(122);plt.imshow(output_image[:,:,::-1]);plt.title("Output Image");plt.axis('off');
        
        # Also Plot the Pose landmarks in 3D.
        mp_drawing.plot_landmarks(results.pose_world_landmarks, mp_pose.POSE_CONNECTIONS)

    else:
        return output_image, landmarks
    
# Read another sample image and perform pose detection on it.
image = cv2.imread('media/sample5.jpg')
detectPose(image, pose, display=True)

# Read another sample image and perform pose detection on it.
image = cv2.imread('media/IMG-20230916-WA0022.jpg')
detectPose(image, pose, display=True)

# Read another sample image and perform pose detection on it.
image = cv2.imread('media/IMG20220925155420.jpg')
detectPose(image, pose, display=True)

# Read another sample image and perform pose detection on it.
image = cv2.imread('media/IMG20220727103338.jpg')
detectPose(image, pose, display=True)



## REAL_TIME FEED / VIDEO ## 
