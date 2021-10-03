import cv2
import numpy as np

mask_img = cv2.imread('videa/mask_dist.jpg')
hadice_img = cv2.imread("videa/hadice.jpg")
THRESHOLD_h = 0.6
THRESHOLD = .5
# cv2.imshow("maska", mask_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# print(mask_img.shape)
w = mask_img.shape[1]
h = mask_img.shape[0]
w_h = hadice_img.shape[1]
h_h = hadice_img.shape[0]
cap = cv2.VideoCapture("Videa/video5.avi")

while True:
    success, img = cap.read()
    crop_vid = img[800:]
    # crop_vid = cv2.cvtColor(crop_vid, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(crop_vid, (3, 3), cv2.BORDER_DEFAULT)
    result = cv2.matchTemplate(crop_vid, mask_img, cv2.TM_CCOEFF_NORMED)
    result_h = cv2.matchTemplate(crop_vid, hadice_img, cv2.TM_CCOEFF_NORMED)
    if len(result):

        yloc, xloc = np.where(result >= THRESHOLD)
        rectangles = []
        for (x, y) in zip(xloc, yloc):
            rectangles.append([int(x), int(y), int(w), int(h)])
        rectangles, weights = cv2.groupRectangles(rectangles, 1, 1)
        for (x, y, w, h) in rectangles:
            cv2.rectangle(crop_vid, (x, y), (x + w, y + h), (0, 255, 255), 2)

        yloc_h, xloc_h = np.where(result_h >= THRESHOLD_h)
        rectangles_h = []
        for (x_h, y_h) in zip(xloc_h, yloc_h):
            rectangles_h.append([int(x_h), int(y_h), int(w_h), int(h_h)])
        rectangles_h, weights_h = cv2.groupRectangles(rectangles_h, 1, 1)
        for (x, y, w_h, h_h) in rectangles_h:
            cv2.rectangle(crop_vid, (x, y), (x + w_h, y + h_h), (0, 0, 255), 2)
    cv2.imshow("Video", crop_vid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break