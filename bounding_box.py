import io
import os
import time
import urllib

import cv2
import numpy as np
import PIL
import requests
import torch
from PIL import Image
from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import check_img_size, non_max_suppression, scale_coords, set_logging
from utils.plots import plot_one_box
from utils.torch_utils import TracedModel
 
def get_model(weights="yolov7.pt", imgsz=256):
    device = "cpu"
    set_logging()
    model = attempt_load(weights, map_location=device)
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size
    model = TracedModel(model, device, imgsz)
    return model, stride, device


def get_prediction(
    img0,
    model,
    stride,
    image_size=256,
    conf_threshold=0.25,
    iou_threshold=0.45,
    augment=False,
    classes=None,
    agnostic_nms=False,
    device="cpu",
):

    img = letterbox(img0, image_size, stride=stride)[0]
    # img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = img.transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device)
    img = img.float()
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    with torch.no_grad():  # Calculating gradients would cause a GPU memory leak
        pred = model(img, augment=augment)[0]
        # Apply NMS
        pred = non_max_suppression(
            pred, conf_threshold, iou_threshold, classes=classes, agnostic=agnostic_nms
        )
    return img, pred


def extract_frame_from_yolo(source, model, stride, device):

    img0 = cv2.imread(source)
    img0 = np.array(img0)

    img, pred = get_prediction(img0, model, stride, image_size=256, device=device)

    # Process detections
    for det in pred:  # detections per image
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                frame = plot_one_box(xyxy, img0)
                cv2.imwrite("pred.jpg", frame)
    print("saved predictions ==================>")

# if __name__ == '__main__':
#     model, stride, device = get_model()
#     extract_frame_from_yolo('Test/Test008/frame_034.jpg', model, stride, device)