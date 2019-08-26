import numpy
import cv2
import pickle


def nothing(x):
    pass


with open("data.pickle", "rb") as handler:
    [objpoints, imgpoints] = pickle.load(handler)

img = cv2.imread('assets/left01.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.namedWindow('bars')
cv2.createTrackbar('alpha', 'bars', 100, 100, nothing)

while True:
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = img.shape[:2]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    alpha = cv2.getTrackbarPos('alpha', 'bars') / 100.0
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), alpha, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    cv2.imshow('calibrated', dst)
    cv2.imshow('original', gray)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(objpoints))

cv2.destroyAllWindows()
