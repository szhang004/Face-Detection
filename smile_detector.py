import cv2
import numpy

while True:
    # Face Classifier
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

    # webcam
    webcam = cv2.VideoCapture(0)
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    # process image
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0, 255, 0), 4)

        # face = (x, y, w, h) - take subimage 
        the_face = frame[y:y+h, x:x+w]

        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # within the face, draw smile
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
            # increases sensitivity to blurriness and nearby detected boxes that match smiles (think haar features)     
        
        #for (x_, y_, w_, h_) in the_face: # nested means variables can't be the same
            #cv2.rectangle(frame, (x_,y_), (x_+w_, y_+h_), (0, 0, 255), 4)

        if len(smiles) > 0:
            cv2.putText(frame, 'smiling', (x,y+h+40), fontScale = 4, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255,255,255))

    # display
    cv2.imshow('Smile Detector', frame)
    key = cv2.waitKey(1)

    if key == 113:
        break


# Cleanup
webcam.release()
cv2.destroyAllWindows()

print("Code Complete")