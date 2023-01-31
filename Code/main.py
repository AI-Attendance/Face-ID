import sys
from time import time, sleep
from multiprocessing import Process, Queue, Value
from queue import Empty
import cv2
import numpy as np

from class_yolo import face_detect
from recognition import face_recognition
from tracker import Tracker, Motion_detect


def ignoreAllQueues() -> None:
    for queue in all_queues:
        queue.cancel_join_thread()
        queue.close()


def recog(quit_flag) -> None:
    face_rec = face_recognition()
    print("Done loading face recognition")
    while not quit_flag.value:
        try:
            objID, selected_face = queue_track_recog.get(
                timeout=options['timeout'])
        except Empty:
            continue
        ret = face_rec.recog(objID,
                             selected_face,
                             align=True,
                             detector_backend='dlib')
        if ret is not None:
            queue_recog_track.put(ret)
        try:
            name = queue_register.get(timeout=options['timeout'])
        except Empty:
            continue
        face_rec.register(name, selected_face)
    ignoreAllQueues()


def yolo(quit_flag) -> None:
    fd = face_detect(kpts=5)
    fd.Load_Prepare_Model()
    print('Done loading yolo')
    while not quit_flag.value:
        try:
            frame = queue_track_yolo.get(timeout=options['timeout'])
        except Empty:
            continue
        rects, conf, cls, kpts = fd.apply_yolo(frame)
        queue_yolo_track.put([rects, conf, cls, kpts])
    ignoreAllQueues()


def track(quit_flag) -> None:
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
    while not quit_flag.value:
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
                # Resize depends on the distance needed to detect faces and on
                # processing speed

        if not queue_yolo_track.empty():
            try:
                rects, conf, cls, kpts = queue_yolo_track.get(
                    timeout=options['timeout'])
            except Empty:
                pass
            else:
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
            try:
                name, ID = queue_recog_track.get(timeout=options['timeout'])
            except Empty:
                pass
            else:
                tr.update_as_known(name, ID)

        # tr.ct.rects contain ids and rectangles
        queue_track_display.put([
            frame, [n['name'] for n in tr.objects.values()],
            [r['rect'] for r in tr.objects.values()]
        ])
    vid.release()
    ignoreAllQueues()


def display(quit_flag) -> None:
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
            quit_flag.value = True
            cv2.destroyAllWindows()
            return
        if key_press == ord('d'):
            print('Saving name...')
            name = input()
            queue_register.put(name)


def exitWithHelpUnless(condition: bool) -> None:
    if condition:
        return
    print("""
          This program runs an attendance system through a camera. It uses AI
          to detect and recognize faces of people. You can invoke the program
          by default arguments or change them to suit your needs.

          Options:

          --timeout __float__   Set the timeout of blocking mechanisms used in
                                multiprocessing queues (default=1.5)
          --yolos __int__       Set the number of yolo subprocesses to run for
                                detection (default=1)
          --recogs __int__      Set the number of recognition subprocesses to
                                run (default=1)
          """)
    exit()


def get_timeout(options: dict) -> None:
    exitWithHelpUnless(
        len(sys.argv) >= 2 and sys.argv[1].replace('.', '', 1).isdigit())
    options[sys.argv[0]] = float(sys.argv[1])
    del sys.argv[:2]


def get_yolos(options: dict) -> None:
    exitWithHelpUnless(len(sys.argv) >= 2 and sys.argv[1].isdigit())
    options[sys.argv[0]] = int(sys.argv[1])
    del sys.argv[:2]


def get_recogs(options: dict) -> None:
    exitWithHelpUnless(len(sys.argv) >= 2 and sys.argv[1].isdigit())
    options[sys.argv[0]] = int(sys.argv[1])
    del sys.argv[:2]


if __name__ == '__main__':
    # load model and prepare everything
    # send inital frame to set image size

    options = {'timeout': 1.5, 'yolos': 1, 'recogs': 1}
    sys.argv.pop(0)
    while len(sys.argv) > 0:
        exitWithHelpUnless(sys.argv[0].startswith('--'))
        sys.argv[0] = sys.argv[0][2:]
        exitWithHelpUnless(sys.argv[0] in options)
        exec('get_' + sys.argv[0] + '(options)')

    queue_track_recog = Queue()
    queue_recog_track = Queue()
    queue_track_display = Queue()
    queue_yolo_track = Queue()
    queue_track_yolo = Queue()
    queue_register = Queue()  # for registering new faces
    all_queues = [
        queue_recog_track, queue_register, queue_track_display,
        queue_track_recog, queue_track_yolo, queue_yolo_track
    ]

    quit_flag = Value('i', False)

    p_track = Process(target=track, args=(quit_flag, ))
    p_yolos = [
        Process(target=yolo, args=(quit_flag, ))
        for _ in range(options['yolos'])
    ]
    p_recogs = [
        Process(target=recog, args=(quit_flag, ))
        for _ in range(options['recogs'])
    ]
    all_processes = [p_track] + p_yolos + p_recogs

    for process in all_processes:
        process.start()

    print('Loading ...')
    display(quit_flag)

    for process in all_processes:
        process.join()

    ignoreAllQueues()
