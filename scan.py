import cv2
import numpy as np
import dlib
import pickle
from concurrent.futures import ThreadPoolExecutor

cap = cv2.VideoCapture(1)  # Use 0 instead of 1 for the default camera

# Set the desired width and height for resizing
desired_width = 400
desired_height = 250
cap.set(3, desired_width)   # Set width
cap.set(4, desired_height)  # Set height

face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

FACE_DESC, FACE_NAME = pickle.load(open('trainset.pk', 'rb'))

# Use ThreadPoolExecutor for multi-threading
executor = ThreadPoolExecutor(max_workers=2)

def recognize_faces(frame, faces):
    descriptors = [model.compute_face_descriptor(frame, sp(frame, det)) for det in faces]
    for (x, y, w, h), face_desc0 in zip(faces, descriptors):
        # Compare with pre-computed descriptors
        distances = np.linalg.norm(FACE_DESC - face_desc0, axis=1)
        idx = np.argmin(distances)

        if distances[idx] < 1:
            name = FACE_NAME[idx]
            print(name)
            cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

def capture_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame
        frame = cv2.resize(frame, (desired_width, desired_height))

        # Detect faces using Haarcascades
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.1, 4)

        # Submit the face recognition task to the thread pool
        executor.submit(recognize_faces, frame, faces)

        cv2.imshow('frame', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

if __name__ == "__main__":
    try:
        capture_frames()
    finally:
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
