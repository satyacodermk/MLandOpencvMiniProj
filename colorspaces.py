# =========================================== ColorSpaces - Object Tracking =======================
"""
-> a color space is a range of colors on a spectrum that can be interpreted and displayed on visual plane 
-> most of the time the display is interpreted through RGB 
-> but as object representation based on RGB data are sensitive to illuminations and shadows
-> for thai we use HSV(Hue-Saturation-Value) color space, that remaps primary colors into dimensions that are
easier for humans to understand.
-> experiments shows that the HSV color algorithm achieved better detection accuracy as compared to RGB colorspace
"""
import cv2    # for processing images and video 
import numpy as np  # to Perform operation on large data
import time    #to get the time from time() to calculate FPS-Frame Per Second

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
    cap = cv2.VideoCapture('C:\\Users\\SATYAM\\Desktop\\MLProj\\Objtrack.mp4') #static video is used for real time pass 0
    #Initialization for calculating FPS (Frame Per Second) 
    prev_frame_time=0
    new_frame_time=0
    
    #============== Use if your system don't have webcam=> app name:- ip webcam ============
    #To connect the mobile camera as webcam using app called ip webcam
    # address="https://25.254.251.144:8080/video"
    # cap.open(address)

    # Define the scaling factor for the images
    scaling_factor = 0.5

    # Keep reading the frames from the webcam
    # until the user hits the 'Esc' key
    while True:
        # Grab the current frame
        frame = get_frame(cap, scaling_factor)

        # Convert the image to HSV colorspace
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Define range of skin color in HSV
        lower = np.array([0, 70, 60])
        upper = np.array([50, 150, 255])

        # Threshold the HSV image to get only skin color
        mask = cv2.inRange(hsv, lower, upper)

        # Bitwise-AND between the mask and original image
        img_bitwise_and = cv2.bitwise_and(frame, frame, mask=mask)

        # Run median blurring
        img_median_blurred = cv2.medianBlur(img_bitwise_and, 5)
        #Display FPS on window
        new_frame_time=time.time() #to get new frame time
        fps=1/(new_frame_time-prev_frame_time)
        prev_frame_time=new_frame_time
        fps="FPS: "+str(int(fps))   #first convert fps to integer and then to string to pass this as text
        cv2.rectangle(frame,(5,40),(150,100),(255,0,0),2)
        cv2.putText(frame,fps,(8,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) 

        # Display the input and output
        cv2.imshow('Input', frame)
        cv2.imshow('Output', img_median_blurred)

        # Check if the user hit the 'Esc' key
        c = cv2.waitKey(5)
        if c == 27:  #27 is code for Esc key and break will help to close the video
            break

    # Close all the windows
    cv2.destroyAllWindows()
