import cv2
import numpy as np


face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("haarcascade_eye.xml")

cap = cv2.VideoCapture(0)


while True:
    _, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        
        eyes = eye_cascade.detectMultiScale(roi_gray)
        i = 0
        for(ex, ey, ew, eh) in eyes:
            i+=1
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)
            if i==2:
                break
    cv2.imshow("Baslik", img)
    k = cv2.waitKey(30) & 0xFF

    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
