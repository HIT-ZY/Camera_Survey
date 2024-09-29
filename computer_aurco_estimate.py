import numpy as np
import time
import cv2
import cv2.aruco as aruco
import math

#相机内参
mtx = np.array([[822.91171133, 0., 213.17260348],
                [0., 809.77476378, 174.75679048],
                [0., 0., 1.]])

#相机畸变参数
dist = np.array([[-0.74972337, 1.98280917, 0.02770972, 0.02523112, -2.8351881]])

#打开摄像头
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_SIMPLEX #font for displaying text (below)

# num = 0
while True:
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    aruco_dict = aruco.Dictionary_get(aruco.DICT_6X6_250)
    parameters = aruco.DetectorParameters_create()

    '''
    detectMarkers(...)
        detectMarkers(image, dictionary[, corners[, ids[, parameters[, rejectedI
        mgPoints]]]]) -> corners, ids, rejectedImgPoints
    '''

    # 检测和确定标记角点的位置
    corners, ids, rejectedImgPoints = aruco.detectMarkers(gray, aruco_dict, parameters=parameters)

    if ids is not None:
        # 估计marker在相机坐标系中的3D位姿
        # 第一个参数：角向量已经检测到的标记角
        # 第二个参数：marker的尺寸，边长
        # 第三个参数：相机的内参矩阵
        # 第四个参数：相机的畸变系数
        # 输出第一个参数：输出marker坐标系到相机坐标系的旋转变换矩阵R
        # 输出第二个参数：输出marker坐标系到相机坐标系的旋转变换矩阵T
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, 0.2, mtx, dist)

        (rvec-tvec).any() #去掉numpy值数组错误

        # aruco.drawAxis(frame, mtx, dist, rvec, tvec, markerLength) #Draw Axis
        # frame: 绘制坐标轴的图像
        # mtx: 相机的内参矩阵
        # dist: 相机的畸变参数；
        # rvec: 旋转向量
        # tvec: 平移向量
        # markerLength: 绘制坐标轴的长度，单位为m
        # aruco.drawDetectedMarkers(frame, corners) #Draw A square around the markers

        for i in range(rvec.shape[0]):
            cv2.drawFrameAxes(frame, mtx, dist, rvec[i, :, :], tvec[i, :, :], 0.03)
            aruco.drawDetectedMarkers(frame, corners)
        # 显示ID，rvec, tvec, 旋转向量和平移向量

        # 位姿解算

        # 距离估计
        # distance = ((tvec[0][0][2] + 0.02) * 0.0254) * 100  # 单位是米
        distance = (tvec[0][0][2])   # 单位是米

        # 角度估计
        # print(rvec)
        # 考虑Z轴（蓝色）的角度
        # 本来正确的计算方式如下，但是由于蜜汁相机标定的问题，实测偏航角度能最大达到104°所以现在×90/104这个系数作为最终角度
        deg = rvec[0][0][2] / math.pi * 180
        # deg=rvec[0][0][2] / math.pi*180*90/104
        # 旋转矩阵到欧拉角
        R = np.zeros((3, 3), dtype=np.float64)
        cv2.Rodrigues(rvec, R)
        sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        if not singular:  # 偏航，俯仰，滚动
            x = math.atan2(R[2, 1], R[2, 2])
            y = math.atan2(-R[2, 0], sy)
            z = math.atan2(R[1, 0], R[0, 0])
        else:
            x = math.atan2(-R[1, 2], R[1, 1])
            y = math.atan2(-R[2, 0], sy)
            z = 0
        # 偏航，俯仰，滚动换成角度
        rx = x * 180.0 / 3.141592653589793
        ry = y * 180.0 / 3.141592653589793
        rz = z * 180.0 / 3.141592653589793
        print("偏航，俯仰，滚动",rx,ry,rz)

        cv2.putText(frame, "Id: " + str(ids), (0, 40), font, 1, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, 'distance:' + str(round(distance, 4)) + str('m'), (0, 110), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'yaw:' + str(rx), (0, 140), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'pitch:' + str(ry), (0, 170), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, 'roll:' + str(rz), (0, 200), font, 1, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(frame, "rvec: " + str(rvec[i, :, :]), (0, 60), font, 0.5, (0, 255, 0), 2, cv2.LINE_AA)
        # cv2.putText(frame, "tvec: " + str(tvec[i, :, :]), (0, 80), font, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    else:
        cv2.putText(frame, "No Ids", (0, 64), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # 显示
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1)

    if key == 27:         # 按esc键退出
        print('ESC break...')
        cap.release()
        cv2.destroyAllWindows()
        break

    # if key == ord(' '):   # 按空格键保存
    # # num = num + 1
    # # filename = "frames_%s.jpg" % num  # 保存一张图像
    #     filename = str(time.time())[:10] + ".jpg"
    #     cv2.imwrite(filename, frame)
