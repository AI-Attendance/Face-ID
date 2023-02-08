from multiprocessing import Process, Queue
import cv2
from class_yolo_openvino import face_detect
from recognition_openvino import face_recognition
from helping_func_class import register_handler, preprocess_face
from tqdm import tqdm
import numpy as np

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
    number_of_pictures = queue_yolo.get()
    for i in range(number_of_pictures):
        org_frame, name = queue_yolo.get()
        frame = cv2.resize(org_frame, (256, 192))
        rects, conf, cls, kpts = fd.apply_yolo(frame)
        s = [2.5, 2.5]  # frame is reshaped 640*480 to 256*192
        if len(rects) > 0:
            rects = (np.array(rects) * np.array([*s, *s])).astype(int)
            for i in range(len(kpts)):
                kpts[i] = np.reshape(kpts[i], (5, 3))[:, 0:2]
        selected_face = preprocess_face(org_frame, rects[0], kpts[0])
        queue_register.put([name, selected_face])

def register() -> None:
    face_rec = face_recognition()
    face_rec.prepare_for_new_db()
    number_of_pictures = queue_register.get()
    for i in tqdm(range(number_of_pictures),
                  desc='Photos Done',
                  ncols=75,
                  unit='Photos'):
        name, selected_face = queue_register.get()
        featurevec = face_rec.recog(selected_face)
        face_rec.add_name_to_db(name, featurevec)
    face_rec.save_db()
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
