from imutils import resize
import cv2
from PIL import Image
from imutils import rotate
import numpy as np
from os import walk

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

class register_handler:
    def __init__(self):
        self.db_path = '../Faces Pictures/'
        pics = []
        name_path = []
        for root_dir, curr_dir, files in walk(self.db_path):
            pics.append(files)
            name_path.append(curr_dir)
        pics.pop(0)
        name_path = name_path[0]
        self.pic_shape = (640, 480)
        self.index = 0
        self.name_pic = []
        for i, n in enumerate(name_path):
            for p in pics[i]:
                self.name_pic.append([n, p])

    def get_number_of_pictures(self):
        return len(self.name_pic)

    def pad_pic(self, picture, required_size):
        # TODO: fill with zeros till reach the wanted size
        padX = required_size[1] - picture.shape[0]
        padY = required_size[0] - picture.shape[1]
        picture = np.pad(picture,
                         [(0, padX), (0, padY), (0,0)],
                         'constant', constant_values=0)
        return picture

    def resize_to(self, picture, required_size):
        # TODO: resize the image with its original
        # aspect ratio till one of dimintions
        # equal one of required image dims 
        aspect_ratio = picture.shape[0] / picture.shape[1]
        if aspect_ratio > (required_size[1] / required_size[0]):
            picture = resize(picture, height=required_size[1])
        else:
            picture = resize(picture, width=required_size[0])
        return picture

    def resize_picture(self, picture):
        picture = self.resize_to(picture, self.pic_shape)
        picture = self.pad_pic(picture, self.pic_shape)
        return picture

    def read_next_picture(self):
        name = self.name_pic[self.index][0]
        picture = Image.open(self.db_path + name + '/' +
                             self.name_pic[self.index][1])
        picture = np.asarray(picture)
        picture = self.resize_picture(picture)
        self.index += 1
        picture = cv2.cvtColor(picture, cv2.COLOR_BGR2RGB)
        return [name, picture]
