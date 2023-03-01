import cv2 as cv
import numpy as np

WIDTH: int = 400
HEIGHT: int = 400
def process_image(width, height):
    ret, img = cam.read()
    image = cv.resize(img, (width, height))
    return image

def setup_camera():
    cam = cv.VideoCapture(0)
    return cam

cam = setup_camera()

while cv.waitKey(1) != ord('q'):
    processed_image = process_image(WIDTH, HEIGHT)
    cv.imshow("img", processed_image)



# After the loop release the cap object
cam.release()
# Destroy all the windows
cv.destroyAllWindows()
