import centroid_tracker
import numpy as np

class Tracker:

    def __init__(self):
        self.ct = centroid_tracker.CentroidTracker(maxDisappeared=15,
                                                   maxDistance=50,
                                                   minNeighbor=50)
        self.objects = {
            0: {
                'center': [0, 0],
                'rect': [0, 0, 0, 0],
                'disappeared': 0,
                'name': 'unkown',
                'kpts': [[0, 0] for _ in range(5)]
            }
        }
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
