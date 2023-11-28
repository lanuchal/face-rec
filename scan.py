import cv2
import numpy as np
import dlib
import pickle
import threading
import tkinter as tk
from PIL import Image, ImageTk

def face_recognition(frame, face_detector, detector, sp, model, FACE_DESC, FACE_NAME):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        img = frame[y-10:y+h+10, x-10:x+w+10][:, :, ::-1]

        dets = detector(img, 1)
        for k, d in enumerate(dets):
            shape = sp(img, d)
            face_desc0 = model.compute_face_descriptor(img, shape, 1)

            d = []
            for face_new in FACE_DESC:
                d.append(np.linalg.norm(face_new - face_desc0))
            d = np.array(d)
            idx = np.argmin(d)
            if d[idx] < 0.6:
                name = FACE_NAME[idx]
                print(name)
                cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 2)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

def update_gui():
    _, frame = cap.read()
    face_recognition(frame, face_detector, detector, sp, model, FACE_DESC, FACE_NAME)
    
    # Convert the frame to RGB format for Tkinter
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)

    # Update the Tkinter label with the new image
    label.imgtk = imgtk
    label.configure(image=imgtk)

    # Schedule the update_gui function to be called again after 10 milliseconds
    root.after(10, update_gui)

# Create Tkinter window
root = tk.Tk()
root.title("Face Recognition")

# Create a label to display the video feed
label = tk.Label(root)
label.pack()

# Create Thread for face recognition
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC, FACE_NAME = pickle.load(open('trainset.pk', 'rb'))

face_thread = threading.Thread(target=update_gui)

# Start the Thread for face recognition
face_thread.start()

# Run the Tkinter main loop
root.mainloop()

# Release resources when closing the window
cap.release()
cv2.destroyAllWindows()
