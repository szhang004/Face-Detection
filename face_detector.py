'''
takes an image and detectors face by 
converting it to grayscale and detecting features
'''

import cv2

# load pretrained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# choose image to detect face in
img = cv2.imread('2faces.jpg')

# convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img) # returns list of rectangles where faces are detected

# print(face_coordinates)

# Draw rectangles
for rectangle in face_coordinates:
    cv2.rectangle(img, ([rectangle[0], rectangle[1]]), ([rectangle[2]+rectangle[0], rectangle[3]+rectangle[1]]), (0,255.0), 2)
    # check documentation to make sure everything is right!



# display image
cv2.imshow('Face Detector', img)
cv2.waitKey(1) # waits 1 msec to switch frames

print("Code Completed")