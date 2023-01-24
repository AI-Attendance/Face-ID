import torch
import numpy as np
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression
from utils.general import scale_coords

class face_detect:
    def __init__(self, weights=['../Models/yolov7-lite-e.pt'],conf_thres=0.35,
                 iou_thres=0.45, kpts=5, device='cpu', classes=[0]):
        """
        yolo face detection
        weights is the pre traind model
            'yolov7-lite-e.pt'
        conf_thres: how confident that face is detected
        iou_thres: if a face is detected mutiple times and they all intersect
                    over iou_thres persentage then it is considered one face
        """
        self.weights = weights
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.kpts = kpts
        self.device = device
        self.classes = classes

    def Load_Prepare_Model(self):
        """
        """
        # Load model
        # load FP32 model
        self.model = attempt_load(self.weights, map_location=self.device)

    def apply_yolo(self, img):
        # Converat
        # BGR to RGB, to bsx3x416x416
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        
        img = torch.from_numpy(img).to(torch.float32)  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        img = img.unsqueeze(0)
        # Inference
        pred = self.model(img)[0]
        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres,
                                   self.iou_thres, classes=self.classes,
                                   kpt_label=self.kpts)
        # Process detections
        rectangles = []
        kpts = []
        confs = []
        clss = []
        for det in pred:  # detections per image
            if len(det):
                # Rescale boxes from img_size to im0 size
                #scale_coords(img.shape[2:], det[:, :4],
                #             frame.shape, kpt_label=False)
                #scale_coords(img.shape[2:], det[:, 6:],
                #             frame.shape, kpt_label=self.kpts, step=3)
                
                # to pass it to next stage
                for i, (*coortens, conf, cls) in enumerate(reversed(det[:, :6])):
                    rectangles.append(np.array(coortens, dtype=int))
                    kpts.append(np.array(det[i, 6:], dtype=int))
                    confs.append(conf.item())
                    clss.append(int(cls))

        return rectangles, confs, clss, kpts
