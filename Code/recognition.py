from deepface.DeepFace import build_model, represent
from deepface.commons.functions import preprocess_face
import numpy as np
from scipy.spatial.distance import cosine
from keras.engine.functional import Functional
from time import time

class face_recognition:
    def __init__(self, model_name = 'Facenet', dbpath = 'facebase', face_threshold=0.30, max_no_lookup=20):
        self.model_name = model_name
        self.path_to_data_base = dbpath
        
        self.name_vec = np.load(file='{}/name_of_fvec.npy'.format(self.path_to_data_base))
        self.fvec = np.load(file='{}/feature_vectors.npy'.format(self.path_to_data_base))
        self.fvecnorms = np.load(file='{}/norm_of_fvec.npy'.format(self.path_to_data_base))

        self.multiple_lookup_unknown = {}
        self.max_no_multiple_lookup = max_no_lookup
        self.not_found = -1
        self.facemodel = build_model(self.model_name)
        print(self.facemodel.make_predict_function(force=True))
        self.face_threshold = face_threshold
        self.unknown_count = -1
    """
    def recog(self, objID, tracker_objID, selected_face, all_IDs, align=False, detector_backend='skip', enforce_detection=False):
        if objID not in all_IDs or selected_face.shape[0] == 0 or selected_face.shape[1] == 0:
            return None
        elif objID not in self.multiple_lookup_unknown:
            self.multiple_lookup_unknown[objID] = 1
        else:
            self.multiple_lookup_unknown[objID] += 1
            if self.multiple_lookup_unknown[objID] > self.max_no_multiple_lookup:
                self.not_found += 1
                self.multiple_lookup_unknown.pop(objID)
                return ['not found {}'.format(self.not_found), objID, tracker_objID]
        
        try:
            face_rep = represent(selected_face, model_name=self.model_name ,model=self.facemodel, enforce_detection=enforce_detection, 
                   align=align, detector_backend=detector_backend, normalization=self.model_name)
        except ValueError:
            return None
        res = 1 - np.abs(np.dot(face_rep, self.fvec.T) / (self.fvecnorms * np.linalg.norm(face_rep)))
        index = res.argmin()
        if res[index] < self.face_threshold:
            return [self.name_vec[index], objID, tracker_objID]
        elif objID[0:3] == 'ID ':
            self.unknown_count += 1
            return ['unknown {}'.format(self.unknown_count), objID, tracker_objID]
    """
    def recog(self, objID, tracker_objID, selected_face, all_IDs, align=False, detector_backend='skip', enforce_detection=False):
        if objID not in all_IDs or selected_face.shape[0] == 0 or selected_face.shape[1] == 0:
            return None
        elif objID not in self.multiple_lookup_unknown:
            self.multiple_lookup_unknown[objID] = 1
        else:
            self.multiple_lookup_unknown[objID] += 1
            if self.multiple_lookup_unknown[objID] > self.max_no_multiple_lookup:
                self.not_found += 1
                self.multiple_lookup_unknown.pop(objID)
                return ['not found {}'.format(self.not_found), objID, tracker_objID]
        selected_face = preprocess_face(selected_face, target_size=self.facemodel.layers[0].input_shape[0][1:3]
                                        ,enforce_detection=enforce_detection, detector_backend=detector_backend, align=align)
        
        tic = time()
        face_rep = self.facemodel.predict(selected_face, verbose='0')[0]
        tac = time()
        print('predict: ', (tac - tic) * 1000)


        res = 1 - np.abs(np.dot(face_rep, self.fvec.T) / (self.fvecnorms * np.linalg.norm(face_rep)))
        index = res.argmin()
        if res[index] < self.face_threshold:
            return [self.name_vec[index], objID, tracker_objID]
        elif objID[0:3] == 'ID ':
            self.unknown_count += 1
            return ['unknown {}'.format(self.unknown_count), objID, tracker_objID]

    def regester(self, name, selected_face, align=True, detector_backend='dlib', enforce_detection=False):
        print('{} is being regestered.'.format(name))

        face_rep = represent(selected_face, model_name=self.model_name ,model=self.facemodel, enforce_detection=enforce_detection, 
                       align=align, detector_backend=detector_backend, normalization=self.model_name)

        if False:
            self.name_vec = np.array([name])
            self.fvec = np.array([face_rep], dtype=np.float32)
            self.fvecnorms = np.array([np.linalg.norm(face_rep)], dtype=np.float32)
        else:   
            self.name_vec = np.array([*self.name_vec, name])
            self.fvec = np.array([*self.fvec, face_rep], dtype=np.float32)
            self.fvecnorms = np.array([*self.fvecnorms, np.linalg.norm(face_rep)], dtype=np.float32)
        
        np.save(file='{}/name_of_fvec'.format(self.path_to_data_base), arr=self.name_vec)
        np.save(file='{}/feature_vectors'.format(self.path_to_data_base), arr=self.fvec)
        np.save(file='{}/norm_of_fvec'.format(self.path_to_data_base), arr=self.fvecnorms)
        
        print('{} is regestered'.format(name))
