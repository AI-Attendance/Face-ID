import sys
import os
from time import time, sleep
from PIL import Image

from multiprocessing import Process, Queue, Value
from multiprocessing.shared_memory import ShareableList
from queue import Empty

import cv2
import numpy as np

from class_yolo_openvino import face_detect

from imutils import rotate
from deepface.commons.functions import extract_faces, normalize_input
from openvino.runtime import Core
from tqdm import tqdm


class face_recognition:

    def __init__(self, model_name='Facenet512', dbpath='facebase'):
        self.model_name = model_name
        self.path_to_data_base = dbpath
        self.name_vec = []
        self.feature_vec = []
        self.feature_norm = []
        ie = Core()
        model = ie.read_model(
            '../Models/{}_openvino/saved_model.xml'.format(model_name))
        self.facemodel = ie.compile_model(model=model, device_name="CPU")
        self.output_layer = self.facemodel.output(0)

    def recog(self, name, selected_face):
        selected_face = extract_faces(
            selected_face,
            (160, 160),  # self.facemodel.layers[0].input_shape[0][1:3],
            detector_backend='skip',
            grayscale=False,
            enforce_detection=False,
            align=True)[0][0]

        face_rep = self.facemodel(selected_face)[self.output_layer][0]
        face_norm = np.linalg.norm(face_rep)

        self.feature_vec.append(face_rep)
        self.feature_norm.append(face_norm)
        self.name_vec.append(name)

    def save_changes(self):
        np.save(file='{}/name_of_fvec'.format(self.path_to_data_base),
                arr=np.array(self.name_vec))
        np.save(file='{}/feature_vectors'.format(self.path_to_data_base),
                arr=np.array(self.feature_vec))
        np.save(file='{}/norm_of_fvec'.format(self.path_to_data_base),
                arr=np.array(self.feature_norm))


class register_handler:

    def __init__(self):
        self.db_path = '../Faces Pictures/'
        pics = []
        name_path = []
        for root_dir, curr_dir, files in os.walk(self.db_path):
            pics.append(files)
            name_path.append(curr_dir)
        pics.pop(0)
        name_path = name_path[0]
        self.pic_shape = (640, 480)
        self.index = 0
        self.name_pic = []
        for i, n in enumerate(name_path):
            for p in pics[i]:
                self.name_pic.append([n, p])

    def get_number_of_pictures(self):
        return len(self.name_pic)

    def pad_pic(self, picture):
        # TODO: fill with zeros till reach the wanted size
        return picture

    def crop_pic(self, picture):
        # TODO: resize the image with its original
        # aspect ratio till one of dimintions
        # equal one of required image dims
        # then pad it with pad_pic
        return picture

    def resize_picture(self, picture):
        if picture.shape[0] < self.pic_shape[0] or picture.shape[
                1] < self.pic_shape[1]:
            picture = self.pad_pic(picture)
        elif picture.shape[0] > self.pic_shape[0] or picture[
                1] > self.pic_shape[1]:
            picture = self.crop_pic(picture)
        return picture

    def read_next_picture(self):
        name = self.name_pic[self.index][0]
        picture = Image.open(self.db_path + name + '/' +
                             self.name_pic[self.index][1])
        picture = np.asarray(picture)
        picture = self.resize_picture(picture)
        self.index += 1
        return [name, picture]


def preprocess_face(frame, rect, kpts):
    # crop
    selected_face = frame[rect[1]:rect[3], rect[0]:rect[2]]
    # align
    right_eye, left_eye, nose = kpts[0:3]
    middle_eye = (left_eye + right_eye) / 2
    angle = middle_eye - nose  # deltaX, deltaY
    angle = -np.arctan(angle[0] / angle[1]) * 180 / np.pi
    padY = int((rect[3] - rect[1]) * 0.02)
    padX = int((rect[2] - rect[0]) * 0.02)
    # instead of black padding, pad with the background then
    # resize to model input size
    rect[0] = rect[0] - padY if rect[0] - padY > 0 else 0
    rect[1] = rect[1] - padX if rect[1] - padX > 0 else 0
    rect[2] = rect[2] + padY if rect[2] + padY < frame.shape[
        1] else frame.shape[1]
    rect[3] = rect[3] + padX if rect[3] + padX < frame.shape[
        0] else frame.shape[0]
    selected_face = rotate(frame[rect[1]:rect[3], rect[0]:rect[2]],
                           angle=angle)
    return selected_face


def read_picture() -> None:
    # get pictures, names
    # resize to 640*480
    rh = register_handler()
    number_of_pictures = rh.get_number_of_pictures()
    queue_yolo.put(number_of_pictures)
    queue_register.put(number_of_pictures)
    for i in range(number_of_pictures):
        name, frame = rh.read_next_picture()
        queue_yolo.put([frame, name])
        # Resize depends on the distance needed to detect faces and on
        # processing speed


def yolo() -> None:
    fd = face_detect(kpts=5)
    number_of_pictures = queue_yolo.get(timeout=2)
    for i in range(number_of_pictures):
        org_frame, name = queue_yolo.get(timeout=2)
        frame = cv2.resize(org_frame, (256, 192))
        rects, conf, cls, kpts = fd.apply_yolo(frame)
        s = [2.5, 2.5]  # frame is reshaped 640*480 to 256*192
        if len(rects) > 0:
            rects = (np.array(rects) * np.array([*s, *s])).astype(int)
            for i in range(len(kpts)):
                kpts[i] = np.reshape(kpts[i], (5, 3))[:, 0:2]
            # kpt_resized = []
            # for kpt in kpts:
            #     kpt_resized.append(np.multiply(kpt, np.array(s)).astype(int))
            # kpts = kpt_resized
        selected_face = preprocess_face(org_frame, rects[0], kpts[0])
        queue_register.put([name, selected_face])


def register() -> None:
    face_rec = face_recognition()
    number_of_pictures = queue_register.get()
    for i in tqdm(range(number_of_pictures),
                  desc='Photos Done',
                  ncols=75,
                  unit='Photos'):
        name, selected_face = queue_register.get()
        face_rec.recog(name, selected_face)
    face_rec.save_changes()
    print('Done')


if __name__ == '__main__':
    queue_register = Queue()
    queue_yolo = Queue()
    all_queues = [queue_register, queue_yolo]

    p_yolos = [Process(target=yolo) for _ in range(1)]
    p_register = [Process(target=register) for _ in range(1)]
    p_read_picture = Process(target=read_picture)

    all_processes = [p_read_picture] + p_yolos + p_register

    for process in all_processes:
        process.start()
    print('Loading ...')

    for process in all_processes:
        process.join()
