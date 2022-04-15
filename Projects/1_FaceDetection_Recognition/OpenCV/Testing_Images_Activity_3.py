# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 05:37:23 2022
@author: bobbala
"""

import cv2

from Training_Model_Activity_2 import faceDetection, put_text

#directory = "Samples"

#*************************** Training Part Start *****************************
# Execute the following functions only once unless there is a change in training data.
# faces,faceID = Activity_2.getImagesAndLabels('Samples')
# cv2.destroyAllWindows()
# face_recognizer = Activity_2.train_classifier(faces,faceID)
# face_recognizer.write('trainingData.yml')


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainingData.yml')

cascadePath = 'Haarcascades/haarcascade_frontalface_default.xml'
faceCascade = cv2.CascadeClassifier(cascadePath);

names = ['None', 'Vanisha', 'Sunil', 'Hrithika']

img = cv2.imread('Testing/frame11.jpg') # Passing the new image of person which is not part of the training

faces,gray_img = faceDetection(img)


for (x,y,w,h) in faces:
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    id, confidence = recognizer.predict(gray_img[y:y+h, x:x+w])

    print("id =",id)
    confidence = int(100-confidence)
    print('confidence = ', int (confidence))

    put_text(img, str(names[id]), x+5, y)

img = cv2.resize(img, (500,500))
cv2.imshow("Img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
