#============================ Face Detection ==============================
"""


"""

#First import all necessary libraries
import cv2
import numpy as np
import time     #to calculate current time for calculating FPS(Frame Per Seconds)

# Load the Haar cascade file
face_cascade = cv2.CascadeClassifier("C:\\Placement Courses\\Machine Learning Image Processing and Computer Vision\\haar-cascade-files-master\\haarcascade_frontalface_default.xml")

# Check if the cascade file has been loaded correctly
if face_cascade.empty():
	raise IOError('Unable to load the face cascade classifier xml file')

# Initialize the video capture object
cap = cv2.VideoCapture('C:\\Users\\SATYAM\\Desktop\\MLProj\\smiledet.mp4')

#Initialization for calculating FPS (Frame Per Second) 
prev_frame_time=0
new_frame_time=0

# Define the scaling factor
scaling_factor = 0.5

# Iterate until the user hits the 'Esc' key
while True:
    # Capture the current frame
    _, frame = cap.read()

    # Resize the frame
    # frame = cv2.resize(frame, None,
    #         fx=scaling_factor, fy=scaling_factor,
    #         interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Run the face detector on the grayscale image
    face_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

    # Draw a rectangle around the face
    for (x,y,w,h) in face_rects:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 3)
    #Display FPS on window
    new_frame_time=time.time() #to get new frame time
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time
    fps="FPS: "+str(int(fps))   #first convert fps to integer and then to string to pass this as text
    cv2.rectangle(frame,(5,40),(150,100),(255,0,0),2)
    cv2.putText(frame,fps,(8,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) 

    # Display the output
    cv2.imshow('Face Detector', frame)

    # Check if the user hit the 'Esc' key 
    c = cv2.waitKey(1) # to quit the loop or break the loop
    if c == 27:
        break

# Release the video capture object
cap.release()

# Close all the windows
cv2.destroyAllWindows()
