import face_recognition
import cv2
import os 

filename1="1.jpg"
filename2="2.jpg"
current_id=0
label_ids={}
person_temp_encoded=[]

base_dir=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(base_dir,"Images")
i=0
for root,dir,files in os.walk(image_dir):
    for file in files:
        person_temp=face_recognition.load_image_file(file)
        path=os.path.join(root,file)
        print(path)
        label=os.path.basename(root).replace(" ","-").lower()
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1
        id_ = label_ids[label]
        person_temp_encoded.append(face_recognition.face_encodings(person_temp)[0])
        

print(label_ids)


person_one=face_recognition.load_image_file(filename1)
person_one_encoded=face_recognition.face_encodings(person_one)[0]

person_two=face_recognition.load_image_file(filename2)
person_two_encoded=face_recognition.face_encodings(person_two)[0]

known_faces=[
    person_temp_encoded  
]

face_name=[]
face_locations=[]
face_encodings=[]

video_capture=cv2.VideoCapture(0)

while True:
    ret,frame=video_capture.read()

    #BGR to RGB color arrays
    color_array= frame[:,:,::-1]

    face_locations=face_recognition.face_locations(color_array)
    face_encodings=face_recognition.face_encodings(color_array,face_locations)
    face_names=[]

    for face_encoding in face_encodings:
        match=face_recognition.compare_faces(person_temp_encoded,face_encoding,tolerance=0.5)
        name=None
        if match[0]:
            name=label_ids
        elif match[1]:
            print(match)
            print(label_ids[1])
         

        
        face_names.append(name)

    #draw the rectangle t=top r=right b=bottom l=left
    for (t,r,b,l),name in zip(face_locations,face_names):
        if not name:
            continue
        
        #(0,0,255) is color code for drawing box around face
        cv2.rectangle(frame,(l,t),(r,b),(0,0,255),2)

        #drawing the label
        cv2.rectangle(frame,(l,b-25),(r,b),(0,0,255),cv2.FILLED)
        font=cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame,name,(l+6,b-6),font,0.5,(255,255,255),1)
    
    cv2.imshow('Video',frame)

    if cv2.waitKey(10)==ord('x'):
        break

video_capture.release()
cv2.destroyAllWindows()

