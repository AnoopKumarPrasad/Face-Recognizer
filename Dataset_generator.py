import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')#to detect features from the source
cam=cv2.VideoCapture(0)#to initialize the video camera object

id=input('enter user id') #get the user ID from the shell input
sampleNo=0 #initialize the counter variable to store the sample number
while(True):
	ret,img=cam.read()

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	for (x,y,w,h) in faces:
		sampleNo=sampleNo+1 #incrementing sample number
		#saving the captured face in the dataset folder
		cv2.imwrite("dataset/User."+str(id)+"."+str(sampleNo)+".jpg",gray[y:y+h, x:x+w])
		cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
		cv2.waitKey(3)
	cv2.imshow('img',gray)
	cv2.waitKey(1)
	cv2.destroyAllWindows()
	if (sampleNo>500): #collects 500 sample images  of a user 
		break
cam.release()

