'''
takes an image and detectors face by 
converting it to grayscale and detecting features
+ a loop to repeat it on each frame 
'''

import cv2

# load pretrained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# capture video from webcam
webcam = cv2.VideoCapture(0)

while True:

    successful_frame_read, frame = webcam.read()  # successful_frame_read is always True

    # convert to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

   
    # Detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) # returns list of rectangles where faces are detected

    # print(face_coordinates)

    # Draw rectangles
    for rectangle in face_coordinates:
        cv2.rectangle(frame, ([rectangle[0], rectangle[1]]), ([rectangle[2]+rectangle[0], rectangle[3]+rectangle[1]]), (0,255.0), 2)
        # check documentation to make sure everything is right!

    # display image
    cv2.imshow('Face Detector', frame)
   
    key = cv2.waitKey(1) # waits 1 milisec
    if key == 113:
        break

print("Code Completed")