"""


"""

import cv2
import numpy as np
import matplotlib.pylab as plt

cap = cv2.VideoCapture('data/video1.avi')

counter = 0
while (cap.isOpened()):
    ret, frame = cap.read()
    if not ret:
        break

    counter += 1

    if not counter % 10:

        with open('data/pictures/frame_{}.npy'.format(counter), 'wb') as f:
            np.save(f, frame)

        cv2.imwrite('data/pictures/frame_{}.jpg'.format(counter), frame)




    if cv2.waitKey(25) & 0xFF == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()