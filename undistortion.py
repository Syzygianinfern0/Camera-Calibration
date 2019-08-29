import numpy as np
import cv2
import pickle


def nothing(x):
    pass


with open("data.pickle", "rb") as handler:
    [objpoints, imgpoints] = pickle.load(handler)

img = cv2.imread('assets/left11.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

cv2.namedWindow('Bars')
cv2.createTrackbar('fx', 'Bars', mtx.astype(int)[0, 0], 750, nothing)
cv2.createTrackbar('fy', 'Bars', mtx.astype(int)[1, 1], 750, nothing)
cv2.createTrackbar('cx', 'Bars', mtx.astype(int)[0, 2], 750, nothing)
cv2.createTrackbar('cy', 'Bars', mtx.astype(int)[1, 2], 750, nothing)


while True:
    mtx = np.array([[cv2.getTrackbarPos('fx', 'Bars'), 0, cv2.getTrackbarPos('cx', 'Bars')],
                    [0, cv2.getTrackbarPos('fy', 'Bars'), cv2.getTrackbarPos('cx', 'Bars')],
                    [0, 0, 1]], dtype=np.float)

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = img.shape[:2]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

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
