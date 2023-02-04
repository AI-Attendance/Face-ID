import cv2
import CentroidTracker
import numpy as np

class Tracker:
    def __init__(self):
        self.ct = CentroidTracker.CentroidTracker(
                maxDisappeared=15,
                maxDistance=50,
                minNeighbor=50)
        self.objects = {0: {'center' : [0, 0],
                            'rect' : [0, 0, 0, 0],
                            'disappeared' : 0,
                            'name' : 'unkown',
                            'kpts' : [[0, 0] for _ in range(5)]}}
        self.ID_sent = {}
        self.ID_resend = 15

    def update_as_known(self, name, ID):
        try:
            self.ct.update_as_known(name, ID)
        except:
            print('KEYERROR')
            

    def update(self, rects, kpts):
        self.objects = self.ct.update(rects, kpts)
        
    def send_to_recognition(self):
        IDs, rects, kpts = [], [], []
        for ID, objdict in self.objects.items():
            if objdict['name'][0:3] == 'unk':
                if ID in self.ID_sent:
                    self.ID_sent[ID] -= 1
                    if self.ID_sent[ID] == 0:
                        self.ID_sent[ID] = self.ID_resend
                        IDs.append(ID)
                        rects.append(objdict['rect'])
                        kpts.append(objdict['kpts'])
                else:
                    self.ID_sent[ID] = self.ID_resend
                    IDs.append(ID)
                    rects.append(objdict['rect'])
                    kpts.append(objdict['kpts'])
                IDs.append(ID)
                rects.append(objdict['rect'])
                kpts.append(objdict['kpts'])
        return IDs, rects, kpts

class Motion_detect:
    def __init__(self, frame, past_frames=5):
        last_frame = cv2.cvtColor(cv2.resize(frame, (160, 120)),
                              cv2.COLOR_BGR2GRAY)
        last_frame = cv2.GaussianBlur(last_frame, (21, 21), 0)
        self.lsts = [last_frame for _ in range(past_frames)]

    def moving(self, frame):
        mframe = cv2.cvtColor(cv2.resize(frame, (160, 120)),
                              cv2.COLOR_BGR2GRAY)
        mframe = cv2.GaussianBlur(mframe, (21, 21), 0)
        diff = cv2.absdiff(mframe, self.lsts[0])
        diff = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY)[1]
        is_moving = np.sum(np.abs(diff)) > 0
        self.lsts.pop(0)
        self.lsts.append(mframe)
        return is_moving
