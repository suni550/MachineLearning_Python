# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 13:00:34 2022
@author: bobbala
"""
import cv2 
import os

# Create different directories for different persons
person_id = int(input("enter person id:"))

if((os.path.isdir('Samples/' + str(person_id)))):
    print("person_id ", person_id, "Already exists; (Re)capture the images")
    pass
else: 
    os.makedirs('Samples/' + str(person_id))
    print("New folder with person_id ", person_id, "is created")

face_classifier = cv2.CascadeClassifier(r"Haarcascades/haarcascade_frontalface_default.xml")

def face_extractor(img):
    faces = face_classifier.detectMultiScale(img,1.3,5)
    for(x,y,w,h) in faces:
        cropped_face = img[y-35:y+h+35, x-35:x+w+35]
        return cropped_face

    return 0

count = 0
cap = cv2.VideoCapture(0)

while True: 
    ret, frame = cap.read()

    if not ret:
        continue

    if face_extractor(frame) is not None:
        count+=1

        face = cv2.resize(face_extractor(frame),(250,250))
        file_name_path = r'Samples/' + str(person_id) + '/' + 'frame' + str(count)+'.jpg'
        cv2.imwrite(file_name_path, face)
 
        cv2.putText(face, str(count), (100,100), cv2.FONT_HERSHEY_COMPLEX,1, (0,255,0), 2)
        cv2.imshow('Face Cropper', face)
    else:
        print("Face not Found")
        pass

    if cv2.waitKey(1) == 32 or count == 100:
        break

cap.release()
cv2.destroyAllWindows()
print("Colleting Samples Completed")






# Activity 4: Create python file to convert video into image
# - Name of the file : PAI-1020A_<name as per IC>_Project_videotoimage.py or .ipynb
# - Import required packages cv2
# - Create while loop to convert video to image until ‘spacebar’ key is pressed
# - Justify why you are using the techniques
# - Prepare one page report for the above Project Activity
