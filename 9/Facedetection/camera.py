import cv2
#pip install opencv-python
#you need to instll opencv AGAIN, YES THIS IS ANOTHER ENVIRONMENT
#YOU CAN install libraries using terminal menu below
# if it doesnt work try this:     pip install --user opencv-python==4.5.5.64
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture(0)  # zero means we want video from webcam

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()  # capturess images frame by frame

    image_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #converrt to grayscale

    detections = face_detector.detectMultiScale(image_gray, minSize=(100, 100), #using cascade detect my face
                                                minNeighbors=5)  #you can tune the parameters here

    # Draw a rectangle around the faces
    for (x, y, w, h) in detections:  #draw rectangle here with parameters
        print(w, h)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # press   q to kill the process
        break

# When everything is done, release the capture
video_capture.release()  #release memory
cv2.destroyAllWindows()