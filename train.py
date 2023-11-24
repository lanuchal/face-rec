import numpy as np
import cv2
import dlib
import os
import pickle

path = './train_data/'
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Load existing data
if os.path.exists('trainset.pk'):
    FACE_DESC, FACE_NAME = pickle.load(open('trainset.pk', 'rb'))
    FACE_DESC = FACE_DESC.tolist()  # Convert numpy.ndarray to list
    FACE_NAME = FACE_NAME.tolist()  # Convert numpy.ndarray to list
else:
    FACE_DESC = []
    FACE_NAME = []

print(FACE_NAME)
for fn in os.listdir(path):
    new_file = fn[:fn.index('_')]
    if new_file not in FACE_NAME:
        if fn.endswith('.jpg') or fn.endswith('.JPG') or fn.endswith('.jpeg') or fn.endswith('.png'):
            img = cv2.imread(os.path.join(path, fn))
            dets = detector(img, 1)
            for k, d in enumerate(dets):
                shape = sp(img, d)
                face_desc = model.compute_face_descriptor(img[:, :, ::-1], shape, 1)  # Corrected typo: 'comute' to 'compute'
                FACE_DESC.append(face_desc)
                print('loading3...', fn)
                FACE_NAME.append(fn[:fn.index('_')])
pickle.dump((np.array(FACE_DESC), np.array(FACE_NAME)), open('trainset.pk', 'wb'))
