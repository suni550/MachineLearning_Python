# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 07:47:55 2022

@author: bobbala
"""
import cv2
import os
import numpy as np

def getImagesAndLabels(directory):
    features=[]
    labels=[]

    for (path, dir, filenames) in os.walk(directory):

        for filename in filenames:
            id=os.path.basename(path) #fetching subdirectory names
            img_path=os.path.join(path,filename)#fetching image path

            if filename.startswith("."):
                print("Skipping system file")#Skipping files that startwith .
                continue

            img = cv2.imread(img_path) # Read each image one by one
            if img is None:
                print("image not captured properly")
                continue

            faces_rect, gray_img = faceDetection(img)

            print("img_path:",img_path)
            print("id:",id)

           # img = drawRectangle(faces_rect, img)

            (x,y,w,h)=faces_rect[0]
            put_text(img, str([id]),x,y)
            roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from grayscale image
            features.append(roi_gray)
            labels.append(int(id))

    return features,labels

def faceDetection(tmp_img):
    gray_img=cv2.cvtColor(tmp_img,cv2.COLOR_BGR2GRAY)#convert color image to grayscale
    face_haar_cascade=cv2.CascadeClassifier('Haarcascades/haarcascade_frontalface_default.xml')#Load haar classifier

    faces=face_haar_cascade.detectMultiScale(gray_img,1.3,5)#detectMultiScale returns rectangles
    print("face detected:",faces)
    return faces,gray_img

def train_classifier(faces,faceID):
    face_recognizer=cv2.face.LBPHFaceRecognizer_create()
    face_recognizer.train(faces,np.array(faceID))
    face_recognizer.write('trainingData.yml')
    return face_recognizer

def drawRectangle(faces, img):
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0,255), 2)
       # cv2.imshow("Image", img)
       # cv2.waitKey(10)
        return img

#FONT_HERSHEY_DUPLEX
def put_text(img, text, x, y):
    font = cv2.FONT_HERSHEY_COMPLEX 
    fontScale = 1
    color = (255,0,0)
    thickness = 2
    cv2.putText(img,text, (x,y),font, fontScale,color,thickness)
    #cv2.putText(img, text, org, fontFace, fontScale, color)

# faces,faceID = getImagesAndLabels('Samples')
# cv2.destroyAllWindows()
# face_recognizer = train_classifier(faces,faceID)

# # save the training data in yml file
# face_recognizer.write('trainingData.yml')

print("Model Trained Sucessfully")
