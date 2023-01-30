from deepface.DeepFace import build_model, represent
import numpy as np

class face_recognition:

    def __init__(self,
                 model_name='Facenet',
                 dbpath='facebase',
                 face_threshold=0.30,
                 max_no_lookup=10):
        self.model_name = model_name
        self.path_to_data_base = dbpath

        self.name_vec = np.load(
            file='{}/name_of_fvec.npy'.format(self.path_to_data_base))
        self.fvec = np.load(
            file='{}/feature_vectors.npy'.format(self.path_to_data_base))
        self.fvecnorms = np.load(
            file='{}/norm_of_fvec.npy'.format(self.path_to_data_base))

        self.facemodel = build_model(self.model_name)
        self.face_threshold = face_threshold

    def recog(self,
              objID,
              selected_face,
              align=True,
              detector_backend='dlib',
              enforce_detection=False):
        if selected_face.shape[0] == 0 or selected_face.shape[1] == 0:
            return None
        
        face_rep = represent(selected_face,
                             model_name=self.model_name,
                             model=self.facemodel,
                             enforce_detection=enforce_detection,
                             detector_backend=detector_backend,
                             align=align)[0]['embedding']

        res = 1 - np.abs(
            np.dot(face_rep, self.fvec.T) /
            (self.fvecnorms * np.linalg.norm(face_rep)))
        index = res.argmin()
        if res[index] < self.face_threshold:
            return [self.name_vec[index], objID]

    def register(self,
                 name,
                 selected_face,
                 align=True,
                 detector_backend='dlib',
                 enforce_detection=False):
        print('{} is being registered.'.format(name))

        face_rep = represent(selected_face,
                             model_name=self.model_name,
                             model=self.facemodel,
                             enforce_detection=enforce_detection,
                             detector_backend=detector_backend,
                             align=align)[0]['embedding']

        if False:
            self.name_vec = np.array([name])
            self.fvec = np.array([face_rep], dtype=np.float32)
            self.fvecnorms = np.array([np.linalg.norm(face_rep)],
                                      dtype=np.float32)
        else:
            self.name_vec = np.array([*self.name_vec, name])
            self.fvec = np.array([*self.fvec, face_rep], dtype=np.float32)
            self.fvecnorms = np.array(
                [*self.fvecnorms, np.linalg.norm(face_rep)], dtype=np.float32)

        np.save(file='{}/name_of_fvec'.format(self.path_to_data_base),
                arr=self.name_vec)
        np.save(file='{}/feature_vectors'.format(self.path_to_data_base),
                arr=self.fvec)
        np.save(file='{}/norm_of_fvec'.format(self.path_to_data_base),
                arr=self.fvecnorms)

        print('{} is registered'.format(name))
