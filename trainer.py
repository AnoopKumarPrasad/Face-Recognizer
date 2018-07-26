import os #we need the os to access the file list in our dataset folder
import cv2
import numpy as np
from PIL import Image

recognizer = cv2.createLBPHFaceRecognizer() #initialize the recognizer
path = 'dataset'

def getImageId(path): #func that grabs the training inages from  the dataset folder and will also get the corresponding Ids from the file name
	ImagePath=[os.path.join(path,s) for s in os.listdir(path)]#listing all the directories from the dataset folder
	#The above code gets the path of each images in the folder
	faces=[] #Store the faces
	Ids=[] #Store the Ids
	for imgPath in ImagePath:
		faceImg=Image.open(imgPath).convert('L') #Loads the image and coverts it into gray scale
		faceNp=np.array(faceImg,'uint8') #converts image into numpy array
		id = int(os.path.split(imgPath)[-1].split(".")[1])#we get the ID we split the image path and take the first from the last part
		faces.append(faceNp)#Extract the faces and append them in the faces list with the Id
		print(id)
		Ids.append(id)
		cv2.imshow("training",faceNp)
		cv2.waitKey(10)

	return Ids, faces
	#function call and feed the data to the recognizer to train
Ids,faces=getImageId(path)
recognizer.train(faces,np.array(Ids))
recognizer.save('test_trainingdata.yml')
cv2.destroyAllWindows()