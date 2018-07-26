import cv2
import numpy as np
from PIL import Image

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#create a cascade classifier using haar cascade
cam = cv2.VideoCapture(0)#creates avideo capture object
rec=cv2.createLBPHFaceRecognizer()#create a recognizer object
rec.load("test_trainingdata.yml")#load the training data
id=0
fontFace = cv2.FONT_HERSHEY_SIMPLEX#font to write the name of the person in the image

fontscale = 1
fontcolor = (255, 255, 255)
while(True):
	ret, img= cam.read() #capture the frames from the camera object
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)#convert the frame into grayscale

	faces = facedetect.detectMultiScale(gray,1.3,5)#detect and extract faces from images
	for (x,y,w,h) in faces:
		cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)
		id,conf=rec.predict(gray[y:y+h,x:x+w])#Recognize the Id of the user
		if(id==8):
			id="Saurav"
		elif(id == 1):
			id = "Upasana"
		elif(id == 3):
			id = "Nayan Sir"
		elif(id == 4):
			id = "Arnab Sir"
		elif(id == 5):
			id = "kabir"
		elif(id == 6):
			id = "Aakangsha"
		elif (id==7):
			id = "Anish"
		else:
			id="unknown"

		
		cv2.putText(img,str(id),(x,y+h),fontFace,fontscale,fontcolor)#Put predicted Id/Name and rectangle on detected face
	cv2.imshow('img',img)
	if(cv2.waitKey(1) ==ord('q')):
		break;
cam.release() #close the camera
cv2.destroyAllWindows() #close all windows