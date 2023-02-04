# TODO: refactor queues and processes
from time import time, sleep
from multiprocessing import Process, Queue, Value
from multiprocessing.shared_memory import ShareableList
from queue import Empty
import cv2
import numpy as np
from class_yolo_openvino import face_detect
from recognition_openvino import face_recognition
from tracker import Tracker, Motion_detect
from imutils import rotate


def recog() -> None:
    face_rec = face_recognition()
    face_rec.load_db()
    idx = 0
    idx_max = 50
    found_before = [None for _ in range(idx_max)]
    give_up_max = 100
    give_up_search = {}
    print("Done loading face recognition")
    while True:
        ret = queue_track_recog.get()
        if ret is None:
            queue_track_recog.put(None)
            break
        objID, selected_face = ret
        if objID in found_before or objID not in lst:
            continue
        face_rep = face_rec.recog(selected_face)
        ret = face_rec.search_db(face_rep)
        if ret is not None:
            queue_recog_track.put([ret, objID])
            found_before[idx] = objID
            idx = (idx + 1) % 50
            continue
        if objID not in give_up_search:
            give_up_search[objID] = 0
            continue
        give_up_search[objID] += 1
        if give_up_search[objID] > give_up_max:
            found_before[idx] = objID
            idx = (idx + 1) % 50
            queue_recog_track.put(['not found', objID])


def yolo() -> None:
    fd = face_detect(kpts=3)
    print('Done loading yolo')
    while True:
        frame = queue_track_yolo.get()
        if frame is None:
            queue_track_yolo.put(None)
            break
        rects, conf, cls, kpts = fd.apply_yolo(frame)
        queue_yolo_track.put([rects, conf, cls, kpts])


# NOTE should we keep it in track as it is the fastest?
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


def track() -> None:
    # input frame must be 3:4 ratio
    vid = cv2.VideoCapture(0)
    ret = False
    frame = None
    while not ret:
        ret, frame = vid.read()
    frame = cv2.flip(frame, 1)
    skip_timer = 0  # time() + 10
    skip_frames_time = 1 / 20  # max FPS of yolo
    tr = Tracker()
    last_face_seen = 0
    frame_size_yolo = (256, 192)
    frame_size_factor = [
        frame.shape[0] / frame_size_yolo[1],
        frame.shape[1] / frame_size_yolo[0]
    ]
    md = Motion_detect(frame, past_frames=5)
    while True:
        rects = []
        kpts = []
        frame = vid.read()[1]
        last_face_seen += 1
        # motion detection
        # if there is no motion don't send frames for yolo nor face recog
        is_moving = md.moving(frame)

        # Remove if not using selfie cam
        frame = cv2.flip(frame, 1)

        if time() - skip_timer > skip_frames_time:
            skip_timer = time()
            if is_moving or last_face_seen < 15:
                queue_track_yolo.put(cv2.resize(frame, frame_size_yolo))
                # Resize depends on the distance needed to detect faces
                # and on processing speed

        if not queue_yolo_track.empty():
            rects, conf, cls, kpts = queue_yolo_track.get()
            if len(rects) > 0:
                rects = (np.array(rects) * np.array(
                    [*frame_size_factor, *frame_size_factor])).astype(int)
                last_face_seen = 0
                for i in range(len(kpts)):
                    kpts[i] = np.reshape(kpts[i], (5, 3))[:, 0:2]

        tr.update(rects, kpts)

        IDs, rects, kpts = tr.send_to_recognition()
        for ID, rect, kpt in zip(IDs, rects, kpts):
            selected_face = preprocess_face(frame, rect, kpt)
            queue_track_recog.put([ID, selected_face])

        # TODO reset the rest of list
        for i, k in enumerate(tr.objects.keys()):
            lst[i] = k
        for i in range(len(tr.objects.keys()), len(lst)):
            lst[i] = None

        if not queue_recog_track.empty():
            name, ID = queue_recog_track.get()
            tr.update_as_known(name, ID)

        # tr.ct.rects contain ids and rectangles
        queue_track_display.put([
            frame, [n['name'] for n in tr.objects.values()],
            [r['rect'] for r in tr.objects.values()],
            [k for k in tr.objects.keys()]
        ])
    vid.release()


def display() -> None:
    colr = [87, 255, 25]  # [random.randint(0, 255) for _ in range(3)]
    while True:
        frame, names, rects, IDs = queue_track_display.get()
        # draw boxes around faces
        for name, rect, ID in zip(names, rects, IDs):
            (x_start, y_start, x_end, y_end) = rect
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), colr, 2)
            cv2.putText(frame, '{} {}'.format(name,
                                              ID), (rect[0], rect[1] - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5, colr, 1)

        cv2.imshow('Test', frame)
        key_press = cv2.waitKey(1)
        if key_press == ord('q'):
            queue_track_recog.put(None)
            queue_track_yolo.put(None)
            cv2.destroyAllWindows()
            return


def exitWithHelpUnless(condition: bool) -> None:
    if condition:
        return
    print("""
          This program runs an attendance system through a camera. It uses AI
          to detect and recognize faces of people. You can invoke the program
          by default arguments or change them to suit your needs.

          Options:

          --yolos __int__       Set the number of yolo subprocesses to run for
                                detection (default=1)
          --recogs __int__      Set the number of recognition subprocesses to
                                run (default=1)
          """)
    exit()


def get_yolos() -> int:
    exitWithHelpUnless(len(sys.argv) >= 1 and sys.argv[0].isdigit())
    return int(sys.argv.pop(0))


def get_recogs() -> int:
    exitWithHelpUnless(len(sys.argv) >= 1 and sys.argv[0].isdigit())
    return int(sys.argv.pop(0))


if __name__ == '__main__':
    import sys

    # load model and prepare everything
    # send inital frame to set image size

    options = {'yolos': 1, 'recogs': 1}
    del sys.argv[0]
    while len(sys.argv) > 0:
        exitWithHelpUnless(sys.argv[0].startswith('--'))
        option = sys.argv.pop(0)[2:]
        exitWithHelpUnless(option in options)
        options[option] = eval('get_' + option + '()')

    # to keep objects ids updated in recognition
    lst = ShareableList([None for _ in range(10)])

    queue_track_recog = Queue()
    queue_recog_track = Queue()
    queue_track_display = Queue()
    queue_yolo_track = Queue()
    queue_track_yolo = Queue()
    all_queues = [
        queue_recog_track, queue_track_display, queue_track_recog,
        queue_track_yolo, queue_yolo_track
    ]

    p_track = Process(target=track)
    p_yolos = [Process(target=yolo) for _ in range(options['yolos'])]
    p_recogs = [Process(target=recog) for _ in range(options['recogs'])]
    all_processes = [p_track] + p_yolos + p_recogs

    for process in all_processes:
        process.start()

    print('Loading ...')
    display()

    for process in all_processes:
        process.join()
    lst.shm.unlink()
    for queue in all_queues:
        queue.cancel_join_thread()
        queue.close()
