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
cv2.createTrackbar('fx', 'Bars', 1000, 2000, nothing)
cv2.createTrackbar('fy', 'Bars', 1000, 2000, nothing)
cv2.createTrackbar('cx', 'Bars', 1000, 2000, nothing)
cv2.createTrackbar('cy', 'Bars', 1000, 2000, nothing)
cv2.createTrackbar('k1', 'Bars', dist.astype(int)[0, 0], 1, nothing)
cv2.createTrackbar('k2', 'Bars', dist.astype(int)[0, 1], 1, nothing)
cv2.createTrackbar('p1', 'Bars', dist.astype(int)[0, 2], 1, nothing)
cv2.createTrackbar('p2', 'Bars', dist.astype(int)[0, 3], 1, nothing)
cv2.createTrackbar('k3', 'Bars', dist.astype(int)[0, 4], 1, nothing)


while True:
    fx = mtx[0, 0] - 10.0 + cv2.getTrackbarPos('fx', 'Bars') / 100.0
    fy = mtx[1, 1] - 10.0 + cv2.getTrackbarPos('fy', 'Bars') / 100.0
    cx = mtx[0, 2] - 10.0 + cv2.getTrackbarPos('cx', 'Bars') / 100.0
    cy = mtx[1, 2] - 10.0 + cv2.getTrackbarPos('cy', 'Bars') / 100.0

    new_mtx = np.array([[fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]],
                   dtype=np.float)
    # dist = np.array([[cv2.getTrackbarPos('k1', 'Bars'), cv2.getTrackbarPos('k2', 'Bars'),
    #                   cv2.getTrackbarPos('p1', 'Bars'), cv2.getTrackbarPos('p2', 'Bars'),
    #                   cv2.getTrackbarPos('k3', 'Bars')]],
    #                 dtype=np.float)

    # ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    h, w = img.shape[:2]

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(new_mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, new_mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]

    cv2.imshow('calibrated', dst)
    cv2.imshow('original', gray)

    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], new_mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error

    print("total error: ", mean_error / len(objpoints))

cv2.destroyAllWindows()
