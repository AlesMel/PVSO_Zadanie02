from time import sleep
from ximea import xiapi
import cv2 as cv
import numpy as np


def setup_camera():
    cam = xiapi.Camera()
    # settings
    cam.open_device()
    cam.set_exposure(1e5)
    cam.set_param("imgdataformat", "XI_RGB32")
    cam.set_param("auto_wb", 1)
    cam.start_acquisition()
    return cam


def process_image(width, height):
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv.resize(image, (width, height))
    return image

cam = setup_camera()

while True:
    processed_image = process_image(400, 300)
    if cv.waitKey(1) == ord('q'):
        break

cam.stop_acquisition()
cam.close_device()
