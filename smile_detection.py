#========================= Smile Detection (using Haar-cascades) ==============

#import the libraries 
import cv2
import numpy as np
import time
#Initialize the face and eye cascade calssifiers 

face_cascade=cv2.CascadeClassifier('C:\\Placement Courses\\Machine Learning Image Processing and Computer Vision\\haar-cascade-files-master\\haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('C:\\Placement Courses\\Machine Learning Image Processing and Computer Vision\\haar-cascade-files-master\\haarcascade_eye_tree_eyeglasses.xml')
smile_cascade=cv2.CascadeClassifier('C:\\Placement Courses\\Machine Learning Image Processing and Computer Vision\\haar-cascade-files-master\\haarcascade_smile.xml')

#define the function that detects the face eye and smile

def detectFES(gray,frame):
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        #Region of Image (ROI) 
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        #get smile feature points
        smiles=smile_cascade.detectMultiScale(roi_gray,1.8,20)

        #Draw rectangle when the smile is detected
        for (sx,sy,sw,sh) in smiles:
            cv2.putText(frame,"Smile Detected....",(50,50),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2,cv2.LINE_AA)
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(0,0,255),2)
    
    return frame



# Starting the video capture
cap=cv2.VideoCapture('C:\\Users\\SATYAM\\Desktop\\MLProj\\TestSmile.mp4') #passing static video

#Initialization for calculating FPS (Frame Per Second) 
prev_frame_time=0
new_frame_time=0

#To connect the mobile camera as webcam using app called ip webcam
# address="https://25.254.251.144:8080/video"
# cap.open(address)

while cap.isOpened():
    #capture cap frame by frame
    _,frame=cap.read()

    
    #Display FPS on window
    new_frame_time=time.time() #to get new frame time
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time
    fps="FPS: "+str(int(fps))   #first convert fps to integer and then to string to pass this as text
    cv2.rectangle(frame,(5,40),(150,100),(255,0,0),2)
    cv2.putText(frame,fps,(8,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) #line_AA=> anti alias line

    #capture image in monochrome i.e. convert frame into gray scale
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #calls the detectFES() function
    get_result=detectFES(gray,frame) # to detect face eyes and smile

    #cv2.putText(frame,fps,(8,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) 
    #display the result on camera feed
    cv2.imshow("Smile Detection",get_result)

    #the control breaks on "q" key is pressed
    if cv2.waitKey(1)& 0xff==ord('q'): #press q to break loop
        break


#Release the capture once all processing is done
cap.release()
cv2.destroyAllWindows()
