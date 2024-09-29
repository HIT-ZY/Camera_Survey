import cv2 as cv
import numpy as np

# 加载用于生成标记的字典
dictionary = cv.aruco.Dictionary_get(cv.aruco.DICT_6X6_250)

# Generate the marker
markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = cv.aruco.drawMarker(dictionary, 99, 200, markerImage, 1)

cv.imwrite("marker99.png", markerImage)
