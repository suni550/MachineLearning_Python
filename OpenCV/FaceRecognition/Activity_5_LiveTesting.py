# -*- coding: utf-8 -*-
"""
Created on Sun Jan 30 08:13:24 2022
@author: bobbala
"""
import cv2

import Activity_2

#This module captures images via webcam and performs face recognition
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

face_recognizer.read('trainingData.yml')
print("Reading of .yml is completed")
name = ['None', 'Vanisha', 'Sunil']

cap=cv2.VideoCapture(0)

while True:
    ret,frame =cap.read()# captures frame and returns boolean value and captured image
    
    faces_detected, gray_img = Activity_2.faceDetection(frame)

    for face in faces_detected:
        (x,y,w,h)=face
        roi_gray=gray_img[y:y+w, x:x+h] # Crop only the face region from gray image
        label,confidence=face_recognizer.predict(roi_gray)#predicting the label and confidence of given image
        confidence = int(100-confidence)
        print("confidence:",confidence)
        print("label:",label)
        
        gray_img = Activity_2.drawRectangle(faces_detected, frame)

        predicted_name=name[label]
        Activity_2.put_text(frame, predicted_name, x+5, y-5)

        if confidence < 40:
             continue

    cv2.imshow('face recognition tutorial ',frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()











# Activity 5: Create python file for detecting image from video
# - Name of the file : PAI-1020A_<name as per IC>_Project_livetesting.py or .ipynb
# - Import required packages Numpy,OS,cv2
# - Read the yml file created in Activity 2
# - Captures frame and returns boolean value and captured image
# - Predicting the label of given image
# - If confidence less than 40 then don't print predicted face text on screen
# - Create while loop to convert video to image until ‘q’ key is pressed













# https://towardsdatascience.com/real-time-face-recognition-an-end-to-end-project-b738bb0f7348