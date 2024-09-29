import cv2
camera = cv2.VideoCapture(0)
i = 1
while i < 50:
    _, frame = camera.read()
    cv2.imwrite("./"+str(i)+'.png', frame, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])
    cv2.imshow('frame', frame)
    i += 1
cv2.destroyAllWindows()