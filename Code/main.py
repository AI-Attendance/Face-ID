from class_yolo import face_detect
from recognition import face_recognition
from time import time, sleep
from multiprocessing import Process, Queue
import cv2
import numpy as np
from subprocess import call
from deep_sort_realtime.deepsort_tracker import DeepSort
from imutils import rotate

def recog():
    face_rec = face_recognition()
    print('done loading face recognition')
    while True:
        objID, tid, selected_face = queue_track_recog.get()
        all_IDs = queue_temp.get().values()
        all_IDs = [i[0] for i in all_IDs]

        tic = time()
        ret = face_rec.recog(objID, tid,selected_face, all_IDs, align=False, detector_backend='skip')
        tac = time()
        print((tac - tic) * 1000)
        if ret is not None:
            queue_recog_track.put(ret)
        if not queue_regester.empty():
            name = queue_regester.get()
            face_rec.regester(name, selected_face)

def yolo():
    fd = face_detect(kpts=5)
    fd.Load_Prepare_Model()
    print('Done Loading yolo')
    while True:
        frame = queue_track_yolo.get()
        rects, conf, cls, kpts = fd.apply_yolo(frame)
        queue_yolo_track.put([rects, conf, cls, kpts])

def kpts_refared_to_face(face , kpts):
    kptret = []
    for kpt in kpts:
        kptret.append([kpt - np.array((face[0], face[1]))])
    print(kptret)
    return kptret

# the main controller of the program flow not just tracker
def track():
    vid = cv2.VideoCapture(0) # ('https://192.168.1.5:8080/video')

    ret = False
    while not ret:
        ret, frame = vid.read()
    skip_timer = 0  # time() + 10
    skip_frames_time = 0.06
    ID_sent = {}
    ID_resend = 10

    ds_tracker = DeepSort(max_age=5)
    track_dict = {}

    last_frame = cv2.cvtColor(cv2.resize(frame, (160, 120)), cv2.COLOR_BGR2GRAY)
    last_frame = cv2.GaussianBlur(last_frame, (21, 21), 0)
    lsts = [last_frame for _ in range(5)]
    is_moving = False
    while True:
        _, frame = vid.read()

        # motion detection
        # if there is no motion don't send frames for yolo nor face recog
        mframe = cv2.resize(frame, (160, 120))
        mframe = cv2.cvtColor(mframe, cv2.COLOR_BGR2GRAY)
        mframe = cv2.GaussianBlur(mframe, (21, 21), 0)
        diff = cv2.absdiff(mframe, lsts[0])
        diff = cv2.threshold(diff, 5, 255, cv2.THRESH_BINARY)[1]
        is_moving = np.sum(diff) > 100
        lsts.pop(0)
        lsts.append(mframe)

        frame = cv2.flip(frame, 1)
        
        if time() - skip_timer > skip_frames_time:
            skip_timer = time()
            if is_moving:
                queue_track_yolo.put(cv2.resize(frame, (256, 192)), timeout=2)
        
        if not queue_yolo_track.empty():
            rects, conf, cls, kpts = queue_yolo_track.get()
            for i in range(len(kpts)):
                kpts[i] = np.reshape(kpts[i], (5, 3))[:, 0:2]
            if len(cls) != 0:
                rec = []
                s = np.array([2.5, 2.5], dtype=np.float32) # frame is reshaped 640*480 to 256*192
                for rect in rects:
                    rec.append([int(rect[0]*s[0]), int(rect[1]*s[1]), # l, r
                            int(rect[2]*s[0]) - int(rect[0]*s[0]), # w
                            int(rect[3]*s[1]) - int(rect[1]*s[1])]) # h
                rects = rec

                kpt_resized = []
                for kpt in kpts:
                    kpt_resized.append(np.multiply(kpt, s).astype(int))
                kpts = kpt_resized

                bbs = [[r, c, s] for r,c,s in zip(rects, conf, kpts)]
                trackers = ds_tracker.update_tracks(bbs, frame=frame)
            else:
                trackers = ds_tracker.update_tracks([[[0, 0, 0, 0], 0, 0]], frame=frame)
            trackersIDS = []
            for track in trackers:
                trackersIDS.append(track.track_id)
                if track.track_id not in track_dict:
                    track_dict[track.track_id] = ['ID {}'.format(track.track_id), [int(i) for i in track.to_ltrb()], track.get_det_class()]
                else:
                    track_dict[track.track_id][1] = [int(i) for i in track.to_ltrb()]
                    track_dict[track.track_id][2] = track.get_det_class()
            tdkey = list(track_dict.keys())
            for tk in tdkey:
                if tk not in trackersIDS:
                    track_dict.pop(tk)

        for tid, (ID, rect, kpts) in track_dict.items():
            if ID[0:3] in ['ID ', 'unk']:
                right_eye, left_eye, nose = kpts[0:3]
                middle_eye = (left_eye + right_eye) / 2
                angle = middle_eye - nose # deltaX, deltaY
                angle = -np.arctan(angle[0] / angle[1]) * 180 / np.pi
                padY = int((rect[3] - rect[1]) * 0.1)
                padX = int((rect[2] - rect[0]) * 0.1)
                selected_face = rotate(np.pad(frame[rect[1]:rect[3], rect[0]:rect[2]], ((padY, padY), (padX, padX), (0, 0))
                                              , 'constant', constant_values=0), angle=angle)
                if ID in ID_sent:
                    ID_sent[ID] -= 1
                    if ID_sent[ID] == 0:
                        ID_sent[ID] = ID_resend
                        queue_track_recog.put([ID, tid, selected_face])
                else:
                    ID_sent[ID] = ID_resend
                    queue_track_recog.put([ID, tid, selected_face])

        if not queue_recog_track.empty():
            name, ID, tid = queue_recog_track.get()
            try:
                track_dict[tid][0] = name
                if name[:3] not in ['ID ', 'unk', 'not']:
                    print(name)
                ID_sent.pop(ID)
            except KeyError:
            # it is excpected to get this error as keys get deleted and 
            # created in run time
                print('\n\nKEYERROR {}, {}\n\n'.format(name, ID))

        # this shall reduce key error in next section 
        # also reduce the number of unneeded look up
        if not queue_temp.empty():
            queue_temp.get()
        queue_temp.put(track_dict)
        # tr.ct.rects contain ids and rectangles
        queue_track_display.put([frame, list(track_dict.values())])


def display():
    colr = [87, 255, 25]  # [random.randint(0, 255) for _ in range(3)]
    while True:
        frame, objects = queue_track_display.get()
        # draw boxes around faces
        for objID, rect, kpts in objects:
            (x_start, y_start, x_end, y_end) = rect
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), colr, 2)
            cv2.putText(frame, objID, (rect[0], rect[1] - 10),
                        cv2.FONT_HERSHEY_TRIPLEX, 0.5,
                        colr, 1)
            for kpt in kpts:
                cv2.circle(frame, (kpt[0], kpt[1]), 1, colr, 2)
        
        cv2.imshow('Test', frame)
        key_press = cv2.waitKey(1)
        if key_press == ord('q'):
            # vid.release()
            cv2.destroyAllWindows()
            break
        elif key_press == ord('d'):
            print('Saving name')
            with open('regestring', mode='r') as file:
                name = file.readline()[:-1]
                queue_regester.put(name)

if __name__ == '__main__':
    # load model and prepare everything
    # send inital frame to set image size

    queue_track_recog = Queue()
    queue_recog_track = Queue()
    queue_track_display = Queue()
    queue_yolo_track = Queue()
    queue_track_yolo = Queue()
    queue_temp = Queue(maxsize=1) # for updating IDs in recog function (there must be a better way!)
    queue_regester = Queue() # for regester new faces 

    p_recog = [Process(target=recog) for _ in range(1)]
    p_yolos = [Process(target=yolo) for _ in range(1)]
    p_display = Process(target=display)
    p_track = Process(target=track)

    for p in p_yolos:
        p.start()
    for p in p_recog:
        p.start()
    p_display.start()
    print('loading')
    sleep(4)
    p_track.start()
    print('To regester new face write the name in "regestring" file, then stand before the camera and press "d"')
    while True:
        if input() == 'q':
            for p in p_yolos:
                p.terminate()
            for p in p_recog:
                p.terminate()
            p_track.terminate()
            p_display.terminate()
            break

    for p in p_recog:
        p.join()
    for p in p_yolos:
        p.join()
    p_track.join()
    p_display.join()
