import cv2
import numpy as np

# realna maska
# mask_img = cv2.imread('videa/testmask1.jpg')
# THRESHOLD = .5

# paint maska
mask_img = cv2.imread('videa/testmask_paint.jpg')
THRESHOLD = .65

# Grayscale, odkomentovat radek ve while
# mask_img = cv2.imread('videa/testmask_paint.jpg')
# mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
# THRESHOLD = .65

# pro pulku masky, dost false positives
# mask_img= mask_img[:, :16]
w = mask_img.shape[1]
h = mask_img.shape[0]
cap = cv2.VideoCapture("Videa/video11.avi")

# cv2.imshow("maska", mask_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

while True:
    success, img = cap.read()
    crop_vid = img[500:800]
    # crop_vid = cv2.cvtColor(crop_vid, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(crop_vid, (3, 3), cv2.BORDER_DEFAULT)
    result = cv2.matchTemplate(crop_vid, mask_img, cv2.TM_CCOEFF_NORMED)
    if len(result):

        yloc, xloc = np.where(result >= THRESHOLD)
        rectangles = []
        for (x, y) in zip(xloc, yloc):
            rectangles.append([int(x), int(y), int(w), int(h)])
        rectangles, weights = cv2.groupRectangles(rectangles, 1, 1)
        crop = 300
        for (x, y, w, h) in rectangles:
            cv2.rectangle(crop_vid, (x, y), (x + w, y + h), (0, 255, 255), 2)
    cv2.imshow("Video", crop_vid)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
