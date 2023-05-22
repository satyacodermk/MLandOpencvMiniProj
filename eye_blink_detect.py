#====================== Eye Detecting and Blink detection =============

"""
->This Project develops a basic understanding of the systems such as driver drowsiness detection,
eye blink locks, eye detection and also the haar cascades (and .xml files) usage with opencv

->Haar Cascades classifiers is an effective object detection method proposed by paul viola
and michael Jones.
"""

#step-1] Import all necessary libraries
import cv2
import numpy as np

#step-2] Initialize the face and eye cascade calssifiers 

face_cascade=cv2.CascadeClassifier('C:\\Placement Courses\\Machine Learning Image Processing and Computer Vision\\haar-cascade-files-master\\haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('C:\\Placement Courses\\Machine Learning Image Processing and Computer Vision\\haar-cascade-files-master\\haarcascade_eye_tree_eyeglasses.xml')


#step-3] Variable store execution state(Boolean variable)
f_read=True

#step-4] Starting the video capture or capturing the webcam 
cap=cv2.VideoCapture('C:\\Users\\SATYAM\\Desktop\\MLProj\\Testeye1.mp4')

#To connect the mobile camera as webcam using app called ip webcam
# address="https://25.254.251.144:8080/video"
# cap.open(address)

ret_val,img=cap.read()

while(ret_val):
    ret_val,img=cap.read()

    #Convert the recorded image to grayscale for Processing make easy
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    #Apply filter to remove noise present in the image
    gray=cv2.bilateralFilter(gray,5,1,1)

    #Detecting the face for region of image i.e. Region of intreset to be fed to eye cascade classifier
    faces=face_cascade.detectMultiScale(gray,1.3,5,minSize=(150,150)) #minSize=(500,500)

    if len(faces)>0:
        for (x,y,w,h) in faces:
            img=cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

            #roi(region of image) is face which is imput to eye calssifier
            roi_face=gray[y:y+h,x:x+w]
            roi_face_clr=img[y:y+h,x:x+w]
            eyes=eye_cascade.detectMultiScale(roi_face,1.3,5,minSize=(50,50))
            #examining the length of eyes object for each eyes
            if len(eyes)>=2:
                #check if program is running for detection or not by conditional statement
                if f_read:
                    cv2.putText(img,"Eye Detected Press 's' to begin",(70,70),cv2.FONT_HERSHEY_PLAIN,3,(0,255,0),2)
                else:
                    cv2.putText(img,"Eye is open",(70,70),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)
            else:
                if f_read:
                    #here ensure that if the eyes are present before starting and if detected then put text as No eyes detected
                    cv2.putText(img,"No eyes Detected",(70,70),cv2.FONT_HERSHEY_PLAIN,2,(0,0,255),2)
                else:
                    #This will print on console and restart the algorithm
                    print("eye blinking detected")
                    #cv2.waitKey(3000)
                    f_read=True
    else:
        cv2.putText(img,"No face detected",(100,90),cv2.FONT_HERSHEY_PLAIN,2,(0,255,0),2)

    #controlling the algorithm with keys
    cv2.imshow("OUTPUT",img)
    a=cv2.waitKey(1)   #wait to take some action by the user

    if a==ord("q"):  #to stop detection
        break
    elif(a==ord("s") and f_read):
        #this will start the detection
        f_read=False

cap.release()
cv2.destroyAllWindows() #to destroy all windows generated    
            

