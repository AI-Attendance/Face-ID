from time import time
from multiprocessing import Process, Queue
import cv2
import numpy as np

from class_yolo import face_detect
from recognition import face_recognition
from tracker import Tracker, Motion_detect


def recog(queue_track_recog, queue_recog_track, queue_register) -> None:
    face_rec = face_recognition()
    print("Done loading face recognition")
    while True:
        ret = queue_track_recog.get()
        if ret is None:
            queue_track_recog.put(None)
            break
        objID, selected_face = ret
        ret = face_rec.recog(objID,
                             selected_face,
                             align=True,
                             detector_backend='dlib')
        if ret is not None:
            queue_recog_track.put(ret)
        if queue_register.empty():
            continue
        name = queue_register.get()
        if name is None:
            queue_register.put(None)
            break
        face_rec.register(name, selected_face)


def yolo(queue_track_yolo, queue_yolo_track) -> None:
    fd = face_detect(kpts=5)
    fd.Load_Prepare_Model()
    print('Done loading yolo')
    while True:
        frame = queue_track_yolo.get()
        if frame is None:
            queue_track_yolo.put(None)
            break
        rects, conf, cls, kpts = fd.apply_yolo(frame)
        queue_yolo_track.put([rects, conf, cls, kpts])


def track(queue_track_yolo, queue_yolo_track, queue_track_recog,
          queue_recog_track, queue_track_display) -> None:
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
        frame = vid.read()[1]

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

        queue_track_display.put([
            frame, [n['name'] for n in tr.objects.values()],
            [r['rect'] for r in tr.objects.values()]
        ])

    vid.release()


def display(queue_track_display, queue_register) -> None:
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
            break
        if key_press == ord('d'):
            print('Saving name...')
            name = input()
            queue_register.put(name)

    cv2.destroyAllWindows()


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

    p_track = Process(target=track,
                      args=(queue_track_yolo, queue_yolo_track,
                            queue_track_recog, queue_recog_track,
                            queue_track_display))
    p_yolos = [
        Process(target=yolo, args=(queue_track_yolo, queue_yolo_track))
        for _ in range(options['yolos'])
    ]
    p_recogs = [
        Process(target=recog,
                args=(queue_track_recog, queue_recog_track, queue_register))
        for _ in range(options['recogs'])
    ]
    all_processes = [p_track] + p_yolos + p_recogs

    for process in all_processes:
        process.start()

    print('Loading ...')
    display(queue_track_display, queue_register)

    for process in all_processes:
        process.join()
