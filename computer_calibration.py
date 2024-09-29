import cv2
import numpy as np
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

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
# objp = np.zeros((6 * 9, 3), np.float32)
# objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)  # 将世界坐标系建在标定板上，所有点的Z坐标全部为0，所以只需要赋值x和y
# objp = 2.9 * objp  # 打印棋盘格一格的边长为2.9cm
# obj_points = []  # 存储3D点
# img_points = []  # 存储2D点
# images = glob.glob("E:/image/*.png")  # 黑白棋盘的图片路径
#
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     size = gray.shape[::-1]
#     ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
#     if ret:
#         obj_points.append(objp)
#         corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1),
#                                     (cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
#         if [corners2]:
#             img_points.append(corners2)
#         else:
#             img_points.append(corners)
#         cv2.drawChessboardCorners(img, (9, 6), corners, ret)  # 记住，OpenCV的绘制函数一般无返回值
#         cv2.waitKey(1)
# _, mtx, dist, _, _ = cv2.calibrateCamera(obj_points, img_points, size, None, None)
#
# # 内参数矩阵
# Camera_intrinsic = {"mtx": mtx, "dist": dist, }