from time import time
from multiprocessing import Process, Queue
from multiprocessing.shared_memory import ShareableList

import cv2
import numpy as np
from class_yolo_openvino import face_detect
from recognition_openvino import face_recognition
from tracker import Tracker
from helping_func_class import register_handler, preprocess_face, Motion_detect

def recog(shared_object_ids: ShareableList, queue_track_recog: Queue,
          queue_recog_track: Queue) -> None:
    face_rec = face_recognition()
    face_rec.load_db()
    idx = 0
    idx_max = 50
    found_before = [None for _ in range(idx_max)]
    give_up_max = 100
    give_up_search = {}
    if __debug__:
        print("Done loading face recognition")
    while True:
        ret = queue_track_recog.get()
        if ret is None:
            queue_track_recog.put(None)
            return
        objID, selected_face = ret
        if objID in found_before or objID not in shared_object_ids:
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
            queue_recog_track.put(["not found", objID])


def yolo(queue_track_yolo: Queue, queue_yolo_track: Queue) -> None:
    fd = face_detect(kpts=3)
    if __debug__:
        print("Done loading yolo")
    while True:
        frame = queue_track_yolo.get()
        if frame is None:
            queue_track_yolo.put(None)
            return
        rects, conf, cls, kpts = fd.apply_yolo(frame)
        queue_yolo_track.put([rects, conf, cls, kpts])

def track(shared_object_ids: ShareableList, queue_display_track: Queue,
          queue_track_display: Queue) -> None:
    skip_timer = 0  # time() + 10
    skip_frames_time = 1 / 20  # max FPS of yolo
    tr = Tracker()
    last_face_seen = 0
    frame_size_yolo = (256, 192)
    frame = queue_display_track.get()
    frame_size_factor = [
        frame.shape[0] / frame_size_yolo[1],
        frame.shape[1] / frame_size_yolo[0]
    ]
    md = Motion_detect(frame, past_frames=5)
    while True:
        rects = []
        kpts = []
        last_face_seen += 1
        # motion detection
        # if there is no motion don't send frames for yolo nor face recog
        is_moving = md.moving(frame)

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
            shared_object_ids[i] = k
        for i in range(len(tr.objects.keys()), len(shared_object_ids)):
            shared_object_ids[i] = None

        if not queue_recog_track.empty():
            name, ID = queue_recog_track.get()
            tr.update_as_known(name, ID)

        queue_track_display.put([
            frame, [n['name'] for n in tr.objects.values()],
            [r['rect'] for r in tr.objects.values()],
            [k for k in tr.objects.keys()]
        ])
        frame = queue_display_track.get()
        if frame is None:
            return


def display() -> None:
    # input frame must be 3:4 ratio
    vid = cv2.VideoCapture(0)
    ret = False
    frame = None
    img_reshape = register_handler()
    while not ret:
        ret, frame = vid.read()
    colr = [87, 255, 25]  # [random.randint(0, 255) for _ in range(3)]
    while True:
        frame = vid.read()[1]
        # Remove if not using selfie cam
        frame = cv2.flip(frame, 1)
        frame = img_reshape.resize_picture(frame)
        queue_display_track.put(frame)
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
            vid.release()
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
    options = {'yolos': 1, 'recogs': 1}
    del sys.argv[0]
    while len(sys.argv) > 0:
        exitWithHelpUnless(sys.argv[0].startswith('--'))
        option = sys.argv.pop(0)[2:]
        exitWithHelpUnless(option in options)
        options[option] = eval('get_' + option + '()')

    # to keep object ids updated in recognition
    shared_object_ids = ShareableList([None for _ in range(10)])

    queue_track_display = Queue()
    queue_display_track = Queue()
    queue_track_recog = Queue()
    queue_recog_track = Queue()
    queue_track_yolo = Queue()
    queue_yolo_track = Queue()

    track_subprocess = Process(target=track,
                               args=(shared_object_ids, queue_display_track,
                                     queue_track_display))
    yolo_subprocesses = [
        Process(target=yolo, args=(queue_track_yolo, queue_yolo_track))
        for _ in range(options['yolos'])
    ]
    recog_subprocesses = [
        Process(target=recog,
                args=(shared_object_ids, queue_track_recog, queue_recog_track))
        for _ in range(options['recogs'])
    ]
    all_subprocesses = [track_subprocess
                        ] + yolo_subprocesses + recog_subprocesses
    for subprocess in all_subprocesses:
        subprocess.start()

    if __debug__:
        print('Loading ...')
    display()

    shared_object_ids.shm.unlink()
    for queue in [queue_display_track, queue_track_recog, queue_track_yolo]:
        queue.put(None)
    for queue in [
            queue_display_track, queue_recog_track, queue_track_display,
            queue_track_recog, queue_track_yolo, queue_yolo_track
    ]:
        queue.cancel_join_thread()
        queue.close()
    for subprocess in all_subprocesses:
        subprocess.join()
