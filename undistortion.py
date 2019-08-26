import numpy
import cv2
import pickle
import glob

with open("data.pickle", "rb") as handler:
    [objpoints, imgpoints] = pickle.load(handler)

images = glob.glob('assets/*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    # undistort
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    file_name = 'outputs\\' + fname.split('\\')[1]
    cv2.imwrite(file_name, dst)
    cv2.imshow('calibrated', dst)
    cv2.imshow('original', gray)
    cv2.waitKey(500)

cv2.destroyAllWindows()
