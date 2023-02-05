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
        model = ie.read_model('../Models/{}_openvino/saved_model.xml'.format(model_name))
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
    
    def prepare_for_new_db(self):
        self.name_vec = []
        self.fvec = []
        self.fvecnorms = []

    def recog(self,
              selected_face):
        if selected_face.shape[0] == 0 or selected_face.shape[1] == 0:
            return None

        selected_face = extract_faces(selected_face,
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
            return self.name_vec[index]
        else:
            return None
    
    def add_name_to_db(self, name, face_rep):
        # only used in registering
        face_norm = np.linalg.norm(face_rep)
        self.fvec.append(face_rep)
        self.fvecnorms.append(face_norm)
        self.name_vec.append(name)

    def save_db(self):
        np.save(file='{}/name_of_fvec'.format(self.path_to_data_base),
                arr=np.array(self.name_vec))
        np.save(file='{}/feature_vectors'.format(self.path_to_data_base),
                arr=np.array(self.fvec))
        np.save(file='{}/norm_of_fvec'.format(self.path_to_data_base),
                arr=np.array(self.fvecnorms))
