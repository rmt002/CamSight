import cv2
import pickle

home = "C:\\Users\\Roshan\\dev\\python\\CamSight\\Cascades\\haarcascade_frontalface_default.xml"
work = "C:\\Users\\roshan.thaliath\\Desktop\\CamSight\\Cascades\\haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(work)
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("model.yml")

labels = {"person_name": 1}
with open("labels.pickle", 'rb') as f:  # readbytes
    labels = pickle.load(f)
    # invert the data coz data is upsidedown
    labels = {v: k for k, v in labels.items()}
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # Convert to greyscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Apply the cascading filter
    faces = faceCascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        id_, conf = recognizer.predict(roi)
        if conf >= 45:  # and conf<=85:
            print(id_)
            print(labels[id_])
            font = cv2.cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x, y), font, 1,
                        color, stroke, cv2.LINE_AA)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(10) == ord('x'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
