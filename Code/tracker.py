import cv2
import CentroidTracker
#import dlib


class Tracker:
    def __init__(self):
        self.ct = CentroidTracker.CentroidTracker(
                maxDisappeared=8,
                maxDistance=80,
                minNeighbor=80)
    
    def clear_rects(self):
        self.rects = []
        self.trackers = []

    def update_as_known(self, name, ID):
        self.ct.update_as_known(name, ID)

    # return dict {ID : Center}
    def objects(self, rects=None):
        if rects is None:
            return self.ct.objects
        else:
            return self.ct.update(rects)
