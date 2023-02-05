from deepface.commons.functions import extract_faces, normalize_input
from deepface.DeepFace import represent
from openvino.runtime import Core
import numpy as np
from time import time


class face_recognition:

    def __init__(self,
                 model_name='Facenet512',
                 dbpath='facebase',
                 face_threshold=0.25):
        self.model_name = model_name
        self.path_to_data_base = dbpath
        ie = Core()
        model = ie.read_model(
            '../Models/{}_openvino/saved_model.xml'.format(model_name))
        self.facemodel = ie.compile_model(model=model, device_name="CPU")
        self.output_layer = self.facemodel.output(0)
        self.face_threshold = face_threshold

    def load_db(self):
        self.name_vec = np.load(
            file='{}/name_of_fvec.npy'.format(self.path_to_data_base))
        self.fvec = np.load(
            file='{}/feature_vectors.npy'.format(self.path_to_data_base))
        self.fvecnorms = np.load(
            file='{}/norm_of_fvec.npy'.format(self.path_to_data_base))

    def recog(self, selected_face):
        if selected_face.shape[0] == 0 or selected_face.shape[1] == 0:
            return None

        selected_face = extract_faces(
            selected_face,
            (160, 160),  # self.facemodel.layers[0].input_shape[0][1:3],
            detector_backend='skip',
            grayscale=False,
            enforce_detection=False,
            align=True)[0][0]
        face_rep = self.facemodel(selected_face)[self.output_layer][0]
        return face_rep

    def search_db(self, face_rep):
        res = 1 - np.abs(
            np.dot(face_rep, self.fvec.T) /
            (self.fvecnorms * np.linalg.norm(face_rep)))
        index = res.argmin()
        if res[index] < self.face_threshold:
            if __debug__:
                print(self.name_vec[index], res[index])
            return self.name_vec[index]
        else:
            if __debug__:
                print('not found', res[index])
            return None
