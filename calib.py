import numpy as np
import cv2 as cv
import glob
import os
import random


# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = 39 * np.mgrid[0:6,0:9].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.
images = glob.glob('*.jpg')

for rand_step in range(20):
    random.shuffle(images)
    for fname in images[:12]:
        img = cv.imread(fname)
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (6,9), None)
    # If found, add object points, image points (after refining them)
        if ret == True:
            objpoints.append(objp)
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            imgpoints.append(corners)
        # Draw and display the corners
            cv.drawChessboardCorners(img, (6,9), corners2, ret)
            ims = cv.resize(img, (960, 540))
            cv.imshow('img', ims)
            cv.waitKey(500)
            ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    print('ITER:', rand_step, '\n', ret, '\n', mtx, '\n', dist, '\n', '__________')

cv.destroyAllWindows()

# np.savetxt('params.txt', mtx) 
# np.savetxt('params.txt', dist) 

images = glob.glob('*.jpg')

if 'Calib_photos' in os.listdir():
    path = os.getcwd()
    os.chdir('Calib_photos')
    for i in os.listdir():
        os.remove(i)
    os.chdir(path)    
else:
    os.mkdir('Calib_photos')
    
   

for sname in images:
    img = cv.imread(sname)
    h,  w = img.shape[:2]
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    # undistort
    dst = cv.undistort(img, mtx, dist, None, newcameramtx)
    # crop the image
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    cv.imwrite('Calib_photos/' + sname, dst)


