import torch
import numpy as np
from utils.general import non_max_suppression
from openvino.runtime import Core
from time import time

class face_detect:
    def __init__(self,
                 weights='../Models/yolov7-lite-e',
                 conf_thres=0.35,
                 iou_thres=0.45,
                 kpts=5,
                 device='cpu',
                 classes=[0]):
        """
        yolo face detection
        weights is the pre traind model
            'yolov7-lite-e.pt'
        conf_thres: how confident that face is detected
        iou_thres: if a face is detected mutiple times and they all intersect
                    over iou_thres persentage then it is considered one face
        """
        ie = Core()
        model = ie.read_model(weights+'/model.xml')
        self.model = ie.compile_model(model=model,
                                      device_name=device.upper())

        self.output_key = list(self.model(np.zeros((1, 3, 192, 256), dtype=np.float32)).keys())[0]

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.kpts = kpts
        self.device = device
        self.classes = classes

    def apply_yolo(self, img):
        # Converat
        # BGR to RGB, to bsx3x416x416
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray([img], dtype=np.float32)
        
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        # Inference
        pred = self.model(img)[self.output_key]
        pred = torch.from_numpy(pred).to(torch.float32)  # uint8 to fp16/32
        # Apply NMS
        pred = non_max_suppression(prediction=pred,
                               conf_thres=self.conf_thres,
                               iou_thres=self.iou_thres,
                               classes=self.classes,
                               kpt_label=self.kpts,
                               nc=1)

        # Process detections
        rectangles = []
        kpts = []
        confs = []
        clss = []
        for det in pred:  # detections per image
            if len(det) == 0:
                continue
            # to pass it to next stage
            for i, (*coortens, conf, cls) in enumerate(reversed(det[:, :6])):
                rectangles.append(np.array(coortens, dtype=int))
                kpts.append(np.array(det[i, 6:], dtype=int))
                confs.append(conf.item())
                clss.append(int(cls))

        return rectangles, confs, clss, kpts
