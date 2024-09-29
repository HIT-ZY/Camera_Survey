import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import math

# 找棋盘格角点
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001) # 阈值
#棋盘格模板规格
w = 8
h = 5
# 世界坐标系中的棋盘格点,例如(0,0,0), (1,0,0), (2,0,0) ....,(8,5,0)，去掉Z坐标，记为二维矩阵
objp = np.zeros((w*h, 3), np.float32)
objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
objp = objp*29  # 29mm

# 储存棋盘格角点的世界坐标和图像坐标对
objpoints = [] # 在世界坐标系中的三维点
imgpoints = [] # 在图像平面的二维点

images = glob.glob('C:/Users/86151/Desktop/gczb/computer/*.png')  #拍摄的十几张棋盘图片所在目录

i = 1
for fname in images:

    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    # 找到棋盘格角点
    ret, corners = cv2.findChessboardCorners(gray, (w, h), None)
    # 如果找到足够点对，将其存储起来
    if ret == True:
        print("i:", i)
        i = i+1

        cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners)
        # 将角点在图像上显示
        cv2.drawChessboardCorners(img, (w, h), corners, ret)
        cv2.namedWindow('findCorners', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('findCorners', 810, 405)
        cv2.imshow('findCorners', img)
        cv2.waitKey(500)
cv2.destroyAllWindows()
# 标定
ret, mtx, dist, rvecs, tvecs = \
    cv2.calibrateCamera(objpoints, imgpoints, size, None, None)


print("ret:", ret)
print("mtx:\n", mtx)     # 内参数矩阵
print("dist:\n", dist)   # 畸变系数
print("rvecs:\n", rvecs)   # 旋转向量  # 外参数
print("tvecs:\n", tvecs)  # 平移向量  # 外参数

obj_points = objp  # 存储3D点
img_points = []  # 存储2D点

# 从摄像头获取视频图像
camera = cv2.VideoCapture(0)

while True:
    _, frame = camera.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    size = gray.shape[::-1]
    ret, corners = cv2.findChessboardCorners(gray, (8, 5), None)
    if ret:  # 画面中有棋盘格
        img_points = np.array(corners)
        cv2.drawChessboardCorners(frame, (8, 5), corners, ret)
        # rvec: 旋转向量 tvec: 平移向量
        _, rvec, tvec = cv2.solvePnP(obj_points, img_points, mtx, dist)  # 解算位姿
        distance = math.sqrt(tvec[0] ** 2 + tvec[1] ** 2 + tvec[2] ** 2)  # 计算距离
        rvec_matrix = cv2.Rodrigues(rvec)[0]  # 旋转向量->旋转矩阵
        proj_matrix = np.hstack((rvec_matrix, tvec))  # hstack: 水平合并
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]  # 欧拉角
        pitch, yaw, roll = eulerAngles[0], eulerAngles[1], eulerAngles[2]
        cv2.putText(frame, "dist: %.2fcm, yaw: %.2f, pitch: %.2f, roll: %.2f" % (distance, yaw, pitch, roll),
                    (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
            break
    else:  # 画面中没有棋盘格
        cv2.putText(frame, "Unable to Detect Chessboard", (20, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                    (0, 0, 255), 3)
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == 27:  # 按ESC键退出
            break
cv2.destroyAllWindows()