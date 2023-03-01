import cv2 as cv
import numpy as np
import glob

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

ROWS = 6
COLS = 8

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((COLS * ROWS, 3), np.float32)
objp[:, :2] = np.mgrid[0:COLS, 0:ROWS].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images.
objpoints = []  # 3d point in real world space
imgpoints = []  # 2d points in image plane.
images = glob.glob('*.jpg')
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
i = 0

# while True:
#     img = process_image(WIDTH, HEIGHT)
#     cv.imshow("img", img)
#     pressed = cv.waitKey(1)
#     if pressed == ord(' '):
#         cv.imwrite("chessboard{0}.jpg".format(i), img)
#         i += 1
#     elif pressed == ord('q'):
#         exit(0)
images = glob.glob('*.jpg')

for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (7,6), None)
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        # Draw and display the corners
        cv.drawChessboardCorners(img, (7,6), corners2, ret)
        cv.imshow('img', img)
        cv.waitKey(500)

# while cv.waitKey(1) != ord('q'):
#     img = process_image(WIDTH, HEIGHT)
#     cv.imshow("img", img)
#
#     gray_image = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     # find corners
#     ret, corners = cv.findChessboardCorners(gray_image, (COLS, ROWS), None)
#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         corners2 = cv.cornerSubPix(gray_image, corners, (11, 11), (-1, -1), criteria)
#         imgpoints.append(corners2)
#         # Draw and display the corners
#         cv.drawChessboardCorners(img, (7, 6), corners2, ret)
#         cv.imshow('img', img)
#         cv.waitKey(500)

cv.destroyAllWindows()

# After the loop release the cap object
cam.release()
# Destroy all the windows
cv.destroyAllWindows()
