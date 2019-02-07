import os
from PIL import Image
import numpy as np
import cv2
import pickle

base_dir=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(base_dir,"Images")
face_cascade=cv2.CascadeClassifier("C:\\Users\\Roshan\\dev\\python\\CamSight\\Cascades\\haarcascade_frontalface_default.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()

current_id=0
label_ids={}
y_labels=[]
x_train=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png")or file.endswith("jpg"):
            path=os.path.join(root,file)
            print(path)
            label=os.path.basename(root).replace(" ","-").lower()
            #print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            #print(label_ids)
            pil_image=Image.open(path).convert("L") #converts to greyscale
            size=(200,200)
            final_image=pil_image.resize(size,Image.ANTIALIAS)
            image_array=np.array(final_image,"uint8")
            #print(image_array)
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=5)

            for (x,y,w,h) in faces:
               roi=image_array[y:y+h,x:x+w]
               x_train.append(roi)
               y_labels.append(id_)

print('FINAL ARRAYS')
#print(y_labels)
#print(x_train)

with open("labels.pickle",'wb')as f: #wb=>writing bytes
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("model.yml")