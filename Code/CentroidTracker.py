# import the necessary packages
from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np


class CentroidTracker:
    def __init__(self, maxDisappeared=50, maxDistance=50,
                 minNeighbor=50):
        # initialize the next unique object ID along with two ordered
        # dictionaries used to keep track of mapping a given object
        # ID to its centroid and number of consecutive frames it has
        # been marked as "disappeared", respectively
        self.nextObjectID = 0
        self.objects = {}
        # {id->int : {'centers' : [x,y], 'rect' : [x,y,x,y], 'disappeared':int, 'name':str}}

        # store the number of maximum consecutive frames a given
        # object is allowed to be marked as "disappeared" until we
        # need to deregister the object from tracking
        self.maxDisappeared = maxDisappeared

        # store the maximum distance between centroids to associate
        # an object -- if the distance is larger than this maximum
        # distance we'll start to mark the object as "disappeared"
        self.maxDistance = maxDistance

        # store the minimum distance between two neighbor centroids
        # if two objects are closer than minimum one of then is deleted
        self.minNeighbor = minNeighbor
    
    def _center_is_rect(self, center, rect, thresh = 10):
        cX = (rect[0] + rect[2]) / 2.0
        cY = (rect[1] + rect[3]) / 2.0
        distance = (cX - center[0]) **2 + (cY - center[1]) ** 2
        distance = distance ** (0.5)
        return True if distance < 10 else False

    # TODO better impelementation
    def rewire_center_rects(self, rects):
        self.rects.clear()
        for rect in rects:
            for ID, center in self.objects.items():
                if self._center_is_rect(center, rect):
                    self.rects[ID] = rect

    def register(self, centroid, rect):
        objectCentroids = [c['center'] for c in self.objects.values()]
        if len(objectCentroids) > 0:
            D_Neighbor = dist.cdist(np.array(objectCentroids),
                                    np.array([centroid]))
            if np.any(D_Neighbor < self.minNeighbor):
                return

        ID = self.nextObjectID
        # when registering an object we use the next available object
        # ID to store the centroid
        self.objects[ID] = {'center':centroid, 
                            'disappeared':0, 
                            'rect':rect, 
                            'name':'unknown'}
        self.nextObjectID += 1

    def deregister(self, objectID):
        # to deregister an object ID we delete the object ID from
        # both of our respective dictionaries
        del self.objects[objectID]

    def update_skip_frames(self):
        for objectID in list(self.objects.keys()):
            self.objects[objectID]['disappeared'] += 1

            # if we have reached a maximum number of consecutive
            # frames where a given object has been marked as
            # missing, deregister it
            if self.objects[objectID]['disappeared'] > self.maxDisappeared:
                self.deregister(objectID)

    def update_as_known(self, name, ID):
        self.objects[ID]['name'] = name

    # TODO more secure hash
    def list_to_number_hash(self, lst):
        res = 0
        for l in lst:
            res += l
            res *= 1000
        return l

    def update(self, rects):
        # check to see if the list of input bounding box rectangles
        # is empty
        if len(rects) == 0:
            # loop over any existing tracked objects and mark them
            # as disappeared
            self.update_skip_frames()
            # return early as there are no centroids or tracking info
            # to update
            return self.objects

        # initialize an array of input centroids for the current frame
        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        rects_center_dict = {}  # {center:rect}

        # loop over the bounding box rectangles
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            # use the bounding box coordinates to derive the centroid
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)
            rects_center_dict[self.list_to_number_hash(inputCentroids[i])] = [startX, startY, endX, endY]
        
        # if we are currently not tracking any objects take the input
        # centroids and register each of them
        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i], rects[i])

        # otherwise, we are currently tracking objects so we need to
        # try to match the input centroids to existing object
        # centroids
        else:
        # grab the set of object IDs and corresponding centroids
            objectIDs = list(self.objects.keys())
            objectCentroids = [c['center'] for c in self.objects.values()]

            # compute the distance between each pair of object
            # centroids and input centroids, respectively -- our
            # goal will be to match an input centroid to an existing
            # object centroid
            D = dist.cdist(np.array(objectCentroids), inputCentroids)

            # in order to perform this matching we must (1) find the
            # smallest value in each row and then (2) sort the row
            # indexes based on their minimum values so that the row
            # with the smallest value as at the *front* of the index
            # list
            rows = D.min(axis=1).argsort()

            # next, we perform a similar process on the columns by
            # finding the smallest value in each column and then
            # sorting using the previously computed row index list
            cols = D.argmin(axis=1)[rows]

            # in order to determine if we need to update, register,
            # or deregister an object we need to keep track of which
            # of the rows and column indexes we have already examined
            usedRows = set()
            usedCols = set()

            # loop over the combination of the (row, column) index
            # tuples
            for (row, col) in zip(rows, cols):
                # if we have already examined either the row or
                # column value before, ignore it
                if row in usedRows or col in usedCols:
                    continue

                # if the distance between centroids is greater than
                # the maximum distance, do not associate the two
                # centroids to the same object
                if D[row, col] > self.maxDistance:
                    continue

                # otherwise, grab the object ID for the current row,
                # set its new centroid, and reset the disappeared
                # counter
                objectID = objectIDs[row]
                self.objects[objectID]['center'] = inputCentroids[col]
                self.objects[objectID]['rect'] = rects_center_dict[self.list_to_number_hash(inputCentroids[col])]
                self.objects[objectID]['disappeared'] = 0
                # name is unchanged

                # indicate that we have examined each of the row and
                # column indexes, respectively
                usedRows.add(row)
                usedCols.add(col)

            # compute both the row and column index we have NOT yet
            # examined
            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            # in the event that the number of object centroids is
            # equal or greater than the number of input centroids
            # we need to check and see if some of these objects have
            # potentially disappeared
            if D.shape[0] >= D.shape[1]:
                # loop over the unused row indexes
                for row in unusedRows:
                    # grab the object ID for the corresponding row
                    # index and increment the disappeared counter
                    objectID = objectIDs[row]
                    self.objects[objectID]['disappeared'] += 1
                    
                    # check to see if the number of consecutive
                    # frames the object has been marked "disappeared"
                    # for warrants deregistering the object
                    if self.objects[objectID]['disappeared'] > self.maxDisappeared:
                        self.deregister(objectID)


            # otherwise, if the number of input centroids is greater
            # than the number of existing object centroids we need to
            # register each new input centroid as a trackable object
            else:
                for col in unusedCols:
                    self.register(inputCentroids[col], 
                                  rects_center_dict[self.list_to_number_hash(inputCentroids[col])])

        # return the set of trackable objects
        return self.objects
