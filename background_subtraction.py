# ====================================== Background subtraction ================================

"""
-> Background subtraction is one of the best known technique to detecting objects 
-> It will compare parts of video to a background image and foreground image 
-> It will helpful to detecting dynamically moving objects from static camera
-> mostly used for object tracking 
"""

import cv2
import numpy as np
import time

# Define a function to get the current frame from the webcam
def get_frame(cap, scaling_factor):
    # Read the current frame from the video capture object
    _, frame = cap.read()

    # Resize the image
    frame = cv2.resize(frame, None, fx=scaling_factor, 
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    return frame

if __name__=='__main__':
    # Define the video capture object
    cap = cv2.VideoCapture('C:\\Users\\SATYAM\\Desktop\\MLProj\\Objtrack.mp4')
    
    #Initialization for calculating FPS (Frame Per Second) 
    prev_frame_time=0
    new_frame_time=0

    #============== Use if your system don't have webcam=> app name:- ip webcam ============
    #To connect the mobile camera as webcam using app called ip webcam
    # address="https://25.254.251.144:8080/video"
    # cap.open(address)

    # Define the background subtractor object
    bg_subtractor = cv2.createBackgroundSubtractorMOG2()
     
    # Define the number of previous frames to use to learn. 
    # This factor controls the learning rate of the algorithm. 
    # The learning rate refers to the rate at which your model 
    # will learn about the background. Higher value for 
    # ‘history’ indicates a slower learning rate. You can 
    # play with this parameter to see how it affects the output.
    history = 100

    # Define the learning rate
    learning_rate = 1.0/history

    # Keep reading the frames from the webcam 
    # until the user hits the 'Esc' key
    while True:
        # Grab the current frame
        frame = get_frame(cap, 0.5)

        # Compute the mask 
        mask = bg_subtractor.apply(frame, learningRate=learning_rate)

        # Convert grayscale image to RGB color image
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        #Display FPS on window
        new_frame_time=time.time() #to get new frame time
        fps=1/(new_frame_time-prev_frame_time)
        prev_frame_time=new_frame_time
        fps="FPS: "+str(int(fps))   #first convert fps to integer and then to string to pass this as text
        cv2.rectangle(frame,(5,40),(150,100),(255,0,0),2)
        cv2.putText(frame,fps,(8,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) 
        
        # Display the images
        cv2.imshow('Input', frame)
        cv2.imshow('Output', mask & frame)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(10)
        if c == 27:
            break

    # Release the video capture object
    cap.release()
    
    # Destroy all the windows to avoid unnecessary usage of memory
    cv2.destroyAllWindows()
