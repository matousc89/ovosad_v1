import cv2
import numpy as np
mask_img = cv2.imread('videa/testmask1.jpg',cv2.COLOR_BGR2GRAY)
w = mask_img.shape[1]
h = mask_img.shape[0]
cap = cv2.VideoCapture("Videa/video11.avi")

while True:
    success,img=cap.read()
    blur = cv2.GaussianBlur(img, (3, 3), cv2.BORDER_DEFAULT)
    result = cv2.matchTemplate(blur, mask_img, cv2.TM_CCOEFF_NORMED)
    if len(result):
        threshold = .5
        yloc, xloc = np.where(result >= threshold)
        rectangles = []
        for (x, y) in zip(xloc, yloc):
            rectangles.append([int(x), int(y), int(w), int(h)])
        rectangles, weights = cv2.groupRectangles(rectangles, 1, 1)
        crop = 300;
        for (x, y, w, h) in rectangles:
            if y > crop:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imshow("Video", img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break