import cv2
# you need to import one more library, opencv-contrib-python
face_detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read("lbph_classifierOG.yml")
width, height = 165, 220  # my own images I resized
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
camera = cv2.VideoCapture(0)

while (True):
    connected, image = camera.read()
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = face_detector.detectMultiScale(image_gray, scaleFactor=1.5, minSize=(30,30))
    for (x, y, w, h) in detections:
        image_face = cv2.resize(image_gray[y:y + w, x:x + h], (width, height))
        cv2.rectangle(image, (x, y), (x + w, y + h), (0,0,255), 2)
        id, confidence = face_recognizer.predict(image_face)
        name = ""
        if id == 1:
            name = 'Ogul'

        cv2.putText(image, name, (x,y +(w+30)), font, 2, (0,0,255))
        cv2.putText(image, str(confidence), (x,y + (h+50)), font, 1, (0,0,255))

    cv2.imshow("Face", image)
    if cv2.waitKey(1) == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()