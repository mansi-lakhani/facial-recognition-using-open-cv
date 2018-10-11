import numpy as np
import cv2

detector= cv2.CascadeClassifier(r'C:/Users/DELL/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
faces=None
img=None
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    i=0
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        for (x,y,w,h) in faces:
            f=img[y:y+h,x:x+w]
            pic=f"{i}.jpg"
            clr_image=cv2.imwrite(pic,f)
            i=i+1
        break

clr_image=cv2.imwrite("p1.jpg",img)
gray_image=cv2.imwrite("p2.jpg",gray)

cap.release()
cv2.destroyAllWindows()

        
