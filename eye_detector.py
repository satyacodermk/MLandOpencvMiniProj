import numpy as np
import cv2
import time

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
#face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

#https://github.com/Itseez/opencv/blob/master/data/haarcascades/haarcascade_eye.xml
#eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

#note:- must specify the absolute path of harcascade placed in your system
#for your system it might be different so specify carefully
face_cascade=cv2.CascadeClassifier('C:\\Placement Courses\\Machine Learning Image Processing and Computer Vision\\haar-cascade-files-master\\haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('C:\\Placement Courses\\Machine Learning Image Processing and Computer Vision\\haar-cascade-files-master\\haarcascade_eye_tree_eyeglasses.xml')

cap = cv2.VideoCapture('C:\\Users\\SATYAM\\Desktop\\MLProj\\Testeye1.mp4')
#Initialization for calculating FPS (Frame Per Second) 
prev_frame_time=0
new_frame_time=0

#To connect the mobile camera as webcam using app called ip webcam
# address="https://25.254.251.144:8080/video"
# cap.open(address)

while 1:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    #Display FPS on window
    new_frame_time=time.time() #to get new frame time
    fps=1/(new_frame_time-prev_frame_time)
    prev_frame_time=new_frame_time
    fps="FPS: "+str(int(fps))   #first convert fps to integer and then to string to pass this as text
    cv2.rectangle(img,(5,40),(150,100),(255,0,0),2)
    cv2.putText(img,fps,(8,80),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),1,cv2.LINE_AA) 

    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

        cv2.putText(img,"FACE DETECTED",(100,150),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) 

        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
            cv2.putText(img,"EYE DETECTED",(100,230),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2,cv2.LINE_AA) 

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == 27: #to break loop press Esc key
        break

cap.release()
cv2.destroyAllWindows()
