# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 05:37:23 2022
@author: bobbala
"""

import cv2
import Activity_2
from Activity_2 import faceDetection, drawRectangle, put_text

directory = "Samples"

# Execute the following functions only once unless there is a change in training data.
# faces,faceID = Activity_2.getImagesAndLabels('Samples')
# cv2.destroyAllWindows()
# face_recognizer = Activity_2.train_classifier(faces,faceID)
# face_recognizer.write('trainingData.yml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainingData.yml')

cascadePath = 'Haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath);

names = ['None', 'Vanisha', 'Sunil']

#img = cv2.imread('Testing/frame75.jpg') # Passing the image not part of training data
img = cv2.imread('Testing/frame81.jpg') # Passing the image not part of training data
#img = cv2.imread('Testing/frame100.jpg') # Passing the image which is part of the training data

faces,gray_img = faceDetection(img)


for (x,y,w,h) in faces:
    id, confidence = recognizer.predict(gray_img[y:y+h, x:x+w])

    confidence = 100- confidence
    print("id =",id)
    print('confidence level: {}', format(confidence))
    if (confidence) <40: 
        continue
    drawRectangle(faces, img)
    put_text(img, str(names[id]), x+5, y)

img = cv2.resize(img, (500,500))
while True:
    cv2.imshow("Img", img)
    if cv2.waitKey(33 ) == ord('q'):
        break

cv2.destroyAllWindows()
