import cv2
from random import randint
import os
from os.path import isdir
from os import makedirs

path_db = '../Faces Pictures'
vid = cv2.VideoCapture(0)

while True:
    _, frame = vid.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow('test', frame)
    k = cv2.waitKey(1)
    if k == ord('s'):
        name = input('Enter a name: ')
        if not isdir(path_db + '/' + name):
            makedirs(path_db + '/' + name)
        cv2.imwrite('{}/{}/{}.jpg'.format(path_db, name, randint(0, 1000000)), frame)
    elif k == ord('q'):
        cv2.destroyWindow('test')
        vid.release()
        break
