from time import time, sleep
from multiprocessing import Process, Queue

import cv2
import numpy as np

from class_yolo import face_detect
from recognition import face_recognition
from tracker import Tracker


def recog():
    face_rec = face_recognition()
    print('Done loading face recognition module')
    while True:
        objID, selected_face = queue_track_recog.get()
        all_IDs = queue_temp.get()
        ret = face_rec.recog(objID,
                             selected_face,
                             all_IDs,
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
    ID_sent = {}
    ID_resend = 5

    last_frame = cv2.cvtColor(cv2.resize(frame, (160, 120)),
                              cv2.COLOR_BGR2GRAY)
    last_frame = cv2.GaussianBlur(last_frame, (21, 21), 0)
    lsts = [last_frame for _ in range(5)]
    while True:
        _, frame = vid.read()

        # motion detection
        # if there is no motion don't send frames for yolo nor face recog

        mframe = cv2.cvtColor(cv2.resize(frame, (160, 120)),
                              cv2.COLOR_BGR2GRAY)
        mframe = cv2.GaussianBlur(mframe, (21, 21), 0)
        diff = cv2.absdiff(mframe, lsts[0])
        diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
        is_moving = np.sum(diff) > 50
        lsts.pop(0)
        lsts.append(mframe)

        # Remove if not using selfie cam
        frame = cv2.flip(frame, 1)

        if time() - skip_timer > skip_frames_time:
            skip_timer = time()
            if is_moving:
                queue_track_yolo.put(cv2.resize(frame, (256, 192)))
                # Resize depends on the distance needed to detect faces and on processing speed

        if not queue_yolo_track.empty():
            rects, conf, cls, kpts = queue_yolo_track.get()
            rec = []
            s = [2.5, 2.5]  # frame is reshaped 640*480 to 256*192
            for rect in rects:
                rec.append([
                    int(rect[0] * s[0]),
                    int(rect[1] * s[1]),
                    int(rect[2] * s[0]),
                    int(rect[3] * s[1])
                ])
            rects = rec

        tr.objects(rects)
        tr.ct.rewire_center_rects(rects)

        for ID, rect in tr.ct.rects.items():
            if ID[0:3] != 'ID ' and ID[0:3] != 'unk':
                continue
            if ID in ID_sent:
                ID_sent[ID] -= 1
                if ID_sent[ID] != 0:
                    continue
                ID_sent[ID] = ID_resend
                queue_track_recog.put(
                    [ID, frame[rect[1]:rect[3], rect[0]:rect[2]]])
            else:
                ID_sent[ID] = ID_resend
                queue_track_recog.put(
                    [ID, frame[rect[1]:rect[3], rect[0]:rect[2]]])

        if not queue_recog_track.empty():
            name, ID = queue_recog_track.get()
            if ID in ID_sent:
                tr.update_as_known(name, ID)
                ID_sent.pop(ID)

        # this shall reduce key error in next section
        # also reduce the number of unneeded look up
        if not queue_temp.empty():
            queue_temp.get()
        queue_temp.put(tr.objects())
        # tr.ct.rects contain ids and rectangles
        queue_track_display.put([frame, tr.ct.rects])


def display():
    colr = [87, 255, 25]  # [random.randint(0, 255) for _ in range(3)]
    while True:
        frame, objects = queue_track_display.get()
        # draw boxes around faces
        for objID, rect in objects.items():
            (x_start, y_start, x_end, y_end) = rect
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), colr, 2)
            cv2.putText(frame, objID, (rect[0], rect[1] - 10),
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
    queue_temp = Queue(
    )  # for updating IDs in recog function (there must be a better way!)
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
