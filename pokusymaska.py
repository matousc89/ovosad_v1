import cv2
import numpy as np
#načtení obrázků
trees_img = cv2.imread('videa/test_image.png', cv2.IMREAD_UNCHANGED)
mask_img = cv2.imread('videa/testmask1.png', cv2.IMREAD_UNCHANGED)

# teoreticky možnost oříznout už zde
# crop=300;
# crop_img=trees_img[crop:]

# zobrazit masku nebo původní obrázek
# cv2.imshow('mask_img', mask_img)
# cv2.waitKey()
# cv2.destroyAllWindows()
#rozmazání podle gausse
blur = cv2.GaussianBlur(trees_img, (3, 3), cv2.BORDER_DEFAULT)
cv2.imshow('Blur', blur)
cv2.waitKey()
cv2.destroyAllWindows()
#matchování podle template je možné vybrat ještě jiné metody, ale tahle funguje
result = cv2.matchTemplate(blur, mask_img, cv2.TM_CCOEFF_NORMED)


cv2.imshow('Result', result)
cv2.waitKey()
cv2.destroyAllWindows()

#velikost masky
w = mask_img.shape[1]
h = mask_img.shape[0]

#kód pro jeden čtverec (shoda)
# min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
# cv2.rectangle(trees_img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 255, 255), 2)
# cv2.imshow('trees', trees_img)
# cv2.waitKey()
# cv2.destroyAllWindows()

#pro více čtverců
threshold = .5
yloc, xloc = np.where(result >= threshold)
#vykreslení všech schod(čtverců)
# for (x, y) in zip(xloc, yloc):
#     cv2.rectangle(trees_img, (x, y), (x + w, y + h), (0,255,255), 2)

#vybrání jen nepřekrývajících se shod
rectangles = []
for (x, y) in zip(xloc, yloc):
    rectangles.append([int(x), int(y), int(w), int(h)])
# print(len(xloc))
rectangles, weights = cv2.groupRectangles(rectangles, 1, 1)
crop = 300;
for (x, y, w, h) in rectangles:
    if y>crop:
        cv2.rectangle(trees_img, (x, y), (x + w, y + h), (0, 255, 255), 2)

cv2.imshow('trees', trees_img)
cv2.waitKey()
cv2.destroyAllWindows()
