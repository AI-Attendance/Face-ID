from time import time, sleep
from multiprocessing import Process, Queue

import cv2
import numpy as np

from class_yolo import face_detect
from recognition import face_recognition
from tracker import Tracker, Motion_detect

def recog():
    face_rec = face_recognition()
    print('Done loading face recognition module')
    while True:
        objID, selected_face = queue_track_recog.get()
        ret = face_rec.recog(objID,
                             selected_face,
                             align=True,
                             detector_backend='dlib')
        if ret is not None:
            queue_recog_track.put(ret)
        if not queue_register.empty():
            name = queue_register.get()
            face_rec.register(name, selected_face)

def yolo():
    fd = face_detect(kpts=5)
    fd.Load_Prepare_Model()
    print('Done loading yolo')
    while True:
        frame = queue_track_yolo.get()
        rects, conf, cls, kpts = fd.apply_yolo(frame)
        queue_yolo_track.put([rects, conf, cls, kpts])

def track():
    vid = cv2.VideoCapture(0)
    ret = False
    frame = None
    while not ret:
        ret, frame = vid.read()
    rects = []
    skip_timer = 0  # time() + 10
    skip_frames_time = 0.05
    tr = Tracker()

    md = Motion_detect(frame, past_frames=5)
    while True:
        _, frame = vid.read()

        # motion detection
        # if there is no motion don't send frames for yolo nor face recog
        is_moving = md.moving(frame)

        # Remove if not using selfie cam
        frame = cv2.flip(frame, 1)

        if time() - skip_timer > skip_frames_time:
            skip_timer = time()
            if is_moving:
                queue_track_yolo.put(cv2.resize(frame, (256, 192)))
                # Resize depends on the distance needed to detect faces and on processing speed

        if not queue_yolo_track.empty():
            rects, conf, cls, kpts = queue_yolo_track.get()
            s = [2.5, 2.5]  # frame is reshaped 640*480 to 256*192
            if len(rects) > 0:
                rects = (np.array(rects) * np.array([*s, *s])).astype(int)
        
        if is_moving:
            tr.update(rects)

        IDs, rects = tr.send_to_recognition()
        for ID, rect in zip(IDs, rects):
            queue_track_recog.put(
                    [ID, frame[rect[1]:rect[3], rect[0]:rect[2]]])

        if not queue_recog_track.empty():
            name, ID = queue_recog_track.get()
            tr.update_as_known(name, ID)

        # tr.ct.rects contain ids and rectangles
        queue_track_display.put([frame,
                                 [n['name'] for n in tr.objects.values()],
                                 [r['rect'] for r in tr.objects.values()]])


def display():
    colr = [87, 255, 25]  # [random.randint(0, 255) for _ in range(3)]
    while True:
        frame, names, rects = queue_track_display.get()
        # draw boxes around faces
        for name, rect in zip(names, rects):
            (x_start, y_start, x_end, y_end) = rect
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), colr, 2)
            cv2.putText(frame, name, (rect[0], rect[1] - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, colr, 1)

        cv2.imshow('Test', frame)
        key_press = cv2.waitKey(1)
        if key_press == ord('q'):
            # vid.release()
            cv2.destroyAllWindows()
            break
        elif key_press == ord('d'):
            print('Saving name')
            with open('registering', mode='r') as file:
                name = file.readline()[:-1]
                queue_register.put(name)


if __name__ == '__main__':
    # load model and prepare everything
    # send inital frame to set image size

    queue_track_recog = Queue()
    queue_recog_track = Queue()
    queue_track_display = Queue()
    queue_yolo_track = Queue()
    queue_track_yolo = Queue()
    #queue_temp = Queue(
    #)  # for updating IDs in recog function (there must be a better way!)
    queue_register = Queue()  # for register new faces

    p_recogs = [Process(target=recog) for _ in range(1)]
    p_yolos = [Process(target=yolo) for _ in range(1)]
    p_display = Process(target=display)
    p_track = Process(target=track)

    for p in p_recogs:
        p.start()
    for p in p_yolos:
        p.start()
    p_display.start()
    print('Loading')
    sleep(3)
    p_track.start()

    if input() == 'q':
        for p in p_yolos:
            p.terminate()
        for p in p_recogs:
            p.terminate()
        p_track.terminate()
        p_display.terminate()

    for p in p_recogs:
        p.join()
    for p in p_yolos:
        p.join()
    p_track.join()
    p_display.join()
