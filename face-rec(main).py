import numpy as np
import cv2
import os

labels=[]

#label for training dataset of person 1
for _ in range(4):
    labels.append(45)

#label for training dataset of person 2
for _ in range(6):
    labels.append(32)
    
#label for training dataset of person 3
for _ in range(5):
    labels.append(20161)

#label for training dataset of person 4
for _ in range(15):
    labels.append(26)

faces=[]
for i in range(1,5):
    sre=f"k{i}.jpg"
    img=cv2.imread(sre)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces.append(img)
for i in range(1,7):
    sre=f"h{i}.jpg"
    img=cv2.imread(sre)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces.append(img)
for i in range(1,6):
    sre=f"d{i}.jpg"
    img=cv2.imread(sre)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces.append(img)
for i in range(1,16):
    sre=f"m{i}.jpg"
    img=cv2.imread(sre)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces.append(img)
print(faces)


face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.train(faces, np.array(labels))

detector= cv2.CascadeClassifier(r'C:/Users/DELL/AppData/Local/Programs/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
faces=None
img=None
while(True):
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector.detectMultiScale(gray, 1.3, 5)
    i=0
    j=0
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        for (x,y,w,h) in faces:
            f=img[y:y+h,x:x+w]
            f=cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            gray_pics=f"{j}.jpg"
            clr_image=cv2.imwrite(gray_pics,f)
            j=j+1
            label= face_recognizer.predict(f)
            print(label)
        break

clr_image=cv2.imwrite("p1.jpg",img)
gray_image=cv2.imwrite("p2.jpg",gray)
cap.release()
cv2.destroyAllWindows()
