import cv2
import CentroidTracker
import numpy as np

class Tracker:
    def __init__(self):
        self.ct = CentroidTracker.CentroidTracker(
                maxDisappeared=5,
                maxDistance=50,
                minNeighbor=50)
        self.objects = {0: {'center' : [0, 0],
                            'rect' : [0, 0, 0, 0],
                            'disappeared' : 0,
                            'name' : 'unkown'}}
        self.ID_sent = {}
        self.ID_resend = 5
        
        #self.multiple_lookup_unknown = {}
        #self.max_no_multiple_lookup = 20
        #self.not_found = -1
    
    def update_as_known(self, name, ID):
        if ID in self.ID_sent:
            self.ct.update_as_known(name, ID)
            self.ID_sent.pop(ID)

    def update(self, rects):
        self.objects = self.ct.update(rects)
        
        # is this really wanted feature?
        #if objID not in self.multiple_lookup_unknown:
        #    self.multiple_lookup_unknown[objID] = 1
        #else:
        #    self.multiple_lookup_unknown[objID] += 1
        #    if self.multiple_lookup_unknown[
        #            objID] > self.max_no_multiple_lookup:
        #        self.not_found += 1
        #        self.multiple_lookup_unknown.pop(objID)
        #        return ['not found {}'.format(self.not_found), objID]

    def send_to_recognition(self):
        IDs, rects = [], []
        for ID, objdict in self.objects.items():
            if objdict['name'][0:3] == 'unk':
                if ID in self.ID_sent:
                    self.ID_sent[ID] -= 1
                    if self.ID_sent[ID] == 0:
                        self.ID_sent[ID] = self.ID_resend
                        IDs.append(ID)
                        rects.append(objdict['rect'])
                else:
                    self.ID_sent[ID] = self.ID_resend
                    IDs.append(ID)
                    rects.append(objdict['rect'])
        return IDs, rects

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
        diff = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)[1]
        is_moving = np.sum(diff) > 0
        self.lsts.pop(0)
        self.lsts.append(mframe)
        return is_moving
