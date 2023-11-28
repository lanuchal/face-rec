import cv2
import numpy as np
import dlib
import pickle
import threading

old_access = ""
old_count = 0

def face_recognition(frame, face_detector, detector, sp, model, FACE_DESC, FACE_NAME):
    global old_access
    global old_count
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        # img = frame[y-10:y+h+10, x-10:x+w+10][:, :, ::-1]
        img = cv2.resize(frame[y-10:y+h+10, x-10:x+w+10], (0, 0), fx=0.5, fy=0.5)

        dets = detector(img, 0)

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
                if old_access != name :
                    old_access = name
                if old_access == name :
                    old_count = old_count + 1
                    if old_count >= 3:
                        old_count = 0
                        print(name, "บันทึกสำเร็จ")
                cv2.putText(frame, name, (x, y-5), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)

def capture_frames(cap, face_detector, detector, sp, model, FACE_DESC, FACE_NAME):
    while True:
        _, frame = cap.read()
        face_recognition(frame, face_detector, detector, sp, model, FACE_DESC, FACE_NAME)
        cv2.imshow('frame', frame)
        key = cv2.waitKey(10)
        if key == 27:  # 27 คือรหัส ASCII ของปุ่ม Esc
            break

# สร้าง Thread สำหรับการทำ face recognition
cap = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
FACE_DESC, FACE_NAME = pickle.load(open('trainset.pk', 'rb'))

face_thread = threading.Thread(target=capture_frames, args=(cap, face_detector, detector, sp, model, FACE_DESC, FACE_NAME))

# เริ่ม Thread สำหรับ face recognition
face_thread.start()

# รอให้ Thread ทำงานเสร็จ
face_thread.join()

# ปิดทรัพยากร
cap.release()
cv2.destroyAllWindows()
