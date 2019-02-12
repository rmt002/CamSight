import face_recognition
import cv2
import os

# Open the input movie file
input_video = cv2.VideoCapture("input.mp4")
length = int(input_video.get(cv2.CAP_PROP_FRAME_COUNT))

# Create an output movie file (resolution/frame rate must match the input video!)
fourcc = cv2.VideoWriter_fourcc('M', 'P', 'E', 'G')
output_video = cv2.VideoWriter('output.avi', fourcc, 25.07, (1280, 720))

current_id = 0
label_ids = {}
person_temp_encoded = []

# Collect and label all the files in the Images folder
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "Images")
i = 0
for root, dir, files in os.walk(image_dir):
    for file in files:
        path = os.path.join(root, file)
        print(path)
        print(file)
        person_temp = face_recognition.load_image_file(path)

        label = os.path.basename(root).replace(" ", "-").lower()
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1
        id_ = label_ids[label]
        person_temp_encoded.append(
            face_recognition.face_encodings(person_temp)[0])

# #Swap Key and Value in the Dictionary
# labels=dict((v,k) for k,v in label_ids.items())
# print(labels)

# face_locations = []
# face_encodings = []
# face_names = []
# frame_number = 0

# while True:

#     ret, frame = input_video.read()
#     frame_number += 1

#     # If video file is out of frames => quit
#     if not ret:
#         break

#     # Convert the image from BGR to RGB
#     rgb_frame = frame[:, :, ::-1]

#     # Find all the faces and face encodings in the current frame of video
#     face_locations = face_recognition.face_locations(rgb_frame)
#     face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

#     face_names = []

#     for face_encoding in face_encodings:
#         # See if the face is a match for the known face(s)
#         match = face_recognition.compare_faces(
#             person_temp_encoded, face_encoding, tolerance=0.50)

#        #Search the match array at every frame for a True value and then index it with the labels array for the person's name
#         name = None
#         if match:
#             for index in range(len(match)):
#                 if match[i]:
#                     name=labels[i]

#         face_names.append(name)

#     # Label the results
#     for (top, right, bottom, left), name in zip(face_locations, face_names):
#         if not name:
#             continue

#         # Drawing the rectangle
#         cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

#         # Drawing the label
#         cv2.rectangle(frame, (left, bottom - 25),
#                       (right, bottom), (0, 0, 255), cv2.FILLED)
#         font = cv2.FONT_HERSHEY_DUPLEX
#         cv2.putText(frame, name, (left + 6, bottom - 6),
#                     font, 0.5, (255, 255, 255), 1)

#     # Write to utput video file frame by frame
#     print("Writing frame {} / {}".format(frame_number, length))
#     output_video.write(frame)

# #Release the webcam
# input_video.release()
# cv2.destroyAllWindows()
