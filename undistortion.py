import numpy
import cv2
import pickle

with open("data.pickle", "rb") as handler:
    [objpoints, imgpoints] = pickle.load(handler)

img = cv2.imread('assets/left12.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

h, w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y + h, x:x + w]
cv2.imshow('calibrated', dst)
cv2.imshow('original', gray)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('calibresult.png', dst)
