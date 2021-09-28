"""


"""
import glob

import cv2
import numpy as np
import matplotlib.pylab as plt


class VideoLoader():

    def __init__(self, file_name):
        self.cap = cv2.VideoCapture(file_name)

    def read_next(self):
        ret, frame = self.cap.read()
        return frame

        #cap.isOpened()


class PictureLoader():

    def __init__(self, folder_path):
        self.pictures = list(glob.glob("{}/*.npy".format(folder_path)))

    def read_next(self):
        frame = np.load(self.pictures[0])
        print(self.pictures[0])
        del self.pictures[0]
        return frame




X_RGB = X_DEPTH = 1280
Y_RGB = 800
Y_DEPTH = 800 #720

VIDEO = True

if VIDEO:
    # video_name = 'data/video1.avi'
    # video_name = 'data/holo4-sikmo-zvrchu-protislun-tam-prostredektraktora.avi'
    video_name = "data/video14.avi"
    reader = VideoLoader(video_name)
else:
    folder_path = "data/set1"
    reader = PictureLoader(folder_path)




while True:
    frame = reader.read_next()

    image_rgb = frame[0:Y_RGB, :]
    image_depth = frame[Y_RGB:, :, 0] * 256 + frame[Y_RGB:, :, 1]

    image_depth = cv2.convertScaleAbs(image_depth, alpha=0.05)
    cv2.imshow('depth', image_depth)
    cv2.imshow("rgb",image_rgb)



    # img_rgb = image_rgb[:,:,1]
    # img_rgb = cv2.equalizeHist(img_rgb)
    # img_rgb = np.clip(img_rgb, 0, 80)


    # template = np.ones((50, 10), dtype="uint8") * 50
    # img_rgb = cv2.matchTemplate(img_rgb, template, cv2.TM_CCORR_NORMED) * 255

    # img_rgb = cv2.Sobel(img_rgb, cv2.CV_64F, 1, 0, ksize=31)



    # img = image_depth.copy()
    # img_orig = img.copy()
    #
    #
    # img = cv2.convertScaleAbs(img, alpha=0.05)
    # img = ~img
    # img[img > 240] = 0
    #
    # ret, img = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    #
    # kernel = np.array([[1, 1, 1],
    #                     [1, 0, 1],
    #                     [1, 1, 1]], np.uint8)
    #
    # img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)





    # kernel = np.ones((5, 5), np.uint8)
    # # ret, img = cv2.threshold(img, 20000, 256*256-1, cv2.THRESH_BINARY)
    # # img = np.clip(img, 30000, 255*256)
    #
    # close_distance = 300
    # img[img < close_distance] = 256*256-1
    # img = cv2.blur(img, (3, 3))

    # img = (img // 256).astype("uint8")
    # img = cv2.equalizeHist(img)

    # img = ~img


    # img[img < far_distance] = 0

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    # img = cv2.erode(img, kernel, iterations=1)
    # img = cv2.blur(img, (5, 5))
    # ret, img = cv2.threshold(img, 200, 255, cv2.THRESH_BINARY)

    # kernel = np.ones((3, 3), np.uint8)


    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    # img = cv2.erode(img, kernel, iterations=1)
    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # img = cv2.dilate(img, kernel, iterations=1)


    # contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # img_orig = img_rgb
    #
    # img = cv2.dilate(img, kernel, iterations=2)
    # img = cv2.erode(img, kernel, iterations=2)


    # mask = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # img_orig = cv2.bitwise_and(img_rgb, mask)

    # img_orig = img_rgb
    #
    # plt.figure(figsize=(20, 10))
    # plt.subplot(221)
    # plt.imshow(img_orig, cmap="gray")
    #
    # plt.subplot(222)
    # plt.hist(img_orig.ravel(), bins=100)
    # plt.xlim(0, 255)
    #
    # plt.subplot(223)
    # plt.imshow(img, cmap="gray")
    #
    # plt.subplot(224)
    # plt.hist(img.ravel(), bins=100)
    # plt.xlim(0, 255)
    #
    # plt.tight_layout()
    # plt.show()





    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

