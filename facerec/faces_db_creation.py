#THIS SCRIPT IS USED TO CREATE THE DATABASE OF FACES

import cv2
NUM_FACES_TO_SAVE = 100
FACE_NAME = 'Giorgio' #'Alessandro'
FOLDER_TYPE = 'validation' #'validation'


# for face detection
face_cascade = cv2.CascadeClassifier('./haarcascades_config/haarcascade_frontalface_default.xml')

# resolution of the webcam
screen_width = 1280       # try 640 if code fails
screen_height = 720

# default webcam
stream = cv2.VideoCapture(0)
num_saved_faces=0
while(True):
    # capture frame-by-frame
    (grabbed, frame) = stream.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # try to detect faces in the webcam
    faces = face_cascade.detectMultiScale(rgb, scaleFactor=1.3, minNeighbors=5)

    # for each faces found
    face_roi = None
    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        color = (0, 255, 255) # in BGR
        stroke = 5
        face_roi = frame[y:y+h, x:x+w].copy()
        cv2.imshow("ROI", frame[y:y+h, x:x+w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, stroke)

    # Display the label
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 255)
    stroke = 2
    cv2.putText(frame, f'({num_saved_faces}/{NUM_FACES_TO_SAVE})', (10, 20), font, 1, color, stroke, cv2.LINE_AA)

    # show the frame
    cv2.imshow("Image", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):    # Press q to break out
        break                  # of the loop
    if key == ord("s"):
        #save the extracted face to database
        if face_roi is not None:
            cv2.imwrite(f"./facedb/{FOLDER_TYPE}/{FACE_NAME}/img{num_saved_faces}.png", face_roi) #frame[fy:fy+fh, fx:fx+fw])
            num_saved_faces += 1
            if num_saved_faces >= NUM_FACES_TO_SAVE:
                break


# cleanup
stream.release()
cv2.waitKey(1)
cv2.destroyAllWindows()
cv2.waitKey(1)
