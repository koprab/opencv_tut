import cv2
import os
import numpy as np
import imutils
import matplotlib.pyplot as plt
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"
image_path = "car_2.jpg"


def display_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def captch_ex(file_name):
    img = cv2.imread(file_name)
    img = cv2.resize(img,(700,600))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # display_image(gray)
    gray = cv2.bilateralFilter(gray, 13, 18, 18)
    edged = cv2.Canny(gray, 50, 250)  # Perform Edge detection
    contours = cv2.findContours(edged.copy(), cv2.RETR_TREE,
                                cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
    screenCnt = None
    for c in contours:
        # approximate the contour
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * peri, True)
        # if our approximated contour has four points, then
        # we can assume that we have found our screen
        if len(approx) == 4:
            screenCnt = approx
            break
    # Masking the part other than the number plate
    mask = np.zeros(gray.shape, np.uint8)
    new_image = cv2.drawContours(mask, [screenCnt], 0, 255, -1, )
    new_image = cv2.bitwise_and(img, img, mask=mask)

    # Now crop
    (x, y) = np.where(mask == 255)
    (topx, topy) = (np.min(x), np.min(y))
    (bottomx, bottomy) = (np.max(x), np.max(y))
    Cropped = gray[topx:bottomx + 1, topy:bottomy + 1]


    display_image(Cropped)

    print(pytesseract.image_to_string(Cropped, config='--psm 6'))

# file_name ='rec_5.png'


captch_ex(image_path)

# img = cv2.imread(image_path)
# print(img.shape)
