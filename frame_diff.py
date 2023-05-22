#=========== Frame Differencing ================
"""
-> it is one of the techinque to capture the moving objects
-> by taking differences between consecutive frames
-> Displaying the differences between the frames
-> used to identify moving objects
"""

import cv2 # for image and video processing 

# Compute the frame differences
def frame_diff(prev_frame, cur_frame, next_frame):
    # Difference between the current frame and the next frame
    diff_frames_1 = cv2.absdiff(next_frame, cur_frame)

    # Difference between the current frame and the previous frame
    diff_frames_2 = cv2.absdiff(cur_frame, prev_frame)

    return cv2.bitwise_and(diff_frames_1, diff_frames_2)

# Define a function to get the current frame from the webcam
def get_frame(cap, scaling_factor):
    # Read the current frame from the video capture object
    _, frame = cap.read()

    # Resize the image
    frame = cv2.resize(frame, None, fx=scaling_factor,
            fy=scaling_factor, interpolation=cv2.INTER_AREA)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    return gray

if __name__=='__main__':
    # Define the video capture object
    cap = cv2.VideoCapture('C:\\Users\\SATYAM\\Desktop\\MLProj\\Objtrack.mp4')

    #============== Use if your system don't have webcam=> app name:- ip webcam ============
    #To connect the mobile camera as webcam using app called ip webcam
    # address="https://25.254.251.144:8080/video"
    # cap.open(address)

    # Define the scaling factor for the images
    scaling_factor = 0.5

    # Grab the current frame
    prev_frame = get_frame(cap, scaling_factor)

    # Grab the next frame
    cur_frame = get_frame(cap, scaling_factor)

    # Grab the frame after that
    next_frame = get_frame(cap, scaling_factor)

    # Keep reading the frames from the webcam
    # until the user hits the 'Esc' key
    while True:
        # Display the frame difference

        cv2.imshow('Object Movement', frame_diff(prev_frame,
                cur_frame, next_frame))

        # Update the variables
        prev_frame = cur_frame
        cur_frame = next_frame

        # Grab the next frame
        next_frame = get_frame(cap, scaling_factor)

        # Check if the user hit the 'Esc' key
        key = cv2.waitKey(10)
        if key == 27:
            break

    # Close all the windows
    cv2.destroyAllWindows()

# note that use the distrubed video or move object your will see some lines that are moving that is the program about 
#object tracking method
#Working fine