import cv2
import imutils
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"

image_url = r'captcha_images\pjLC.jpg'


def display_image(img):
    cv2.imshow("Image", img)
    cv2.waitKey(0)


def process_image(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # thresh = cv2.threshold(gray,100,250,cv2.THRESH_BINARY| cv2.THRESH_OTSU)[1]
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    display_image(thresh)
    # horizontal_inv = cv2.bitwise_not(thresh)
    # img = cv2.bitwise_and(image,image,mask=th)
    # display_image(horizontal_inv)
    # kernel = np.ones((5, 3), np.uint8)
    # erosion = cv2.erode(horizontal_inv,kernel,iterations=1)
    # display_image(erosion)
    # horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    # detected_noise = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,horizontal_kernel,iterations=2)
    # display_image(detected_noise)
    # kernel_ver = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 1))
    # img = cv2.dilate(thresh,kernel_ver,iterations=2)
    # display_image(img)
    # Remove horizontal
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
    detected_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
    cnts = cv2.findContours(detected_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        cv2.drawContours(detected_lines, [c], -1, (255, 0, 0), 1)
    display_image(detected_lines)
    img = cv2.dilate(detected_lines, horizontal_kernel, iterations=2)
    second_kernel = np.ones((2, 3), np.uint8)
    img = cv2.erode(img, second_kernel, iterations=3)
    display_image(img)
    img = cv2.dilate(img,np.ones((2,1),np.uint8),iterations=2)
    display_image(img)
    # repair_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
    # result = 255 - cv2.morphologyEx(255 - image, cv2.MORPH_CLOSE, repair_kernel, iterations=1)
    # display_image(result)


def contour_filter(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    display_image(thresh)
    kernel = np.ones((7,3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
    display_image(opening)
    closing = cv2.dilate(opening, np.ones((1, 2), np.uint8), iterations=2)
    display_image(closing)
    inverted = cv2.bitwise_not(closing)
    display_image(inverted)
    ctrs, hier = cv2.findContours(inverted, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for i, ctr in enumerate(sorted_ctrs):
        (x, y, w, h) = cv2.boundingRect(ctr)
        print((x, y, w, h))
        roi = inverted[y:y + h, x:x + w]
        display_image(roi)
        area = w * h
        rect = cv2.rectangle(inverted, (x, y), (x + w, y + h), (0,255,0), 1)
        # file_name = f'extracted_letter_images\cont_{i}.png'
        # cv2.imwrite(file_name, roi)
        display_image(rect)
    # for c in cnts:
    #     cv2.drawContours(inverted, [c], -1, (0, 0, 255), 1)
    # display_image(inverted)

def filter_noise(img):
    image = cv2.imread(img)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    display_image(thresh)
    kernel = np.ones((6, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations=1)
    display_image(opening)
    closing = cv2.erode(opening, np.ones((2,1),np.uint8),iterations=5)
    display_image(closing)
    repair_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 1))
    result = cv2.morphologyEx(closing, cv2.MORPH_OPEN,repair_kernel,iterations=2)
    display_image(result)
    repair_kernel_v = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 3))
    result_v = cv2.morphologyEx(result, cv2.MORPH_CLOSE, repair_kernel_v, iterations=2)
    display_image(result_v)
    inverted = cv2.bitwise_not(result)
    inverted = cv2.medianBlur(inverted, 3)
    display_image(inverted)
    print(pytesseract.image_to_string(inverted, config='--psm 12'))


def bitwise_filter(img):
    image = cv2.imread(img, 0)

    display_image(image)
    thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    display_image(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_DILATE, (6, 2))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel , iterations=1)
    display_image(thresh)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, np.ones((3,2),np.uint8), iterations=2)
    display_image(thresh)
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 1))
    # dilate = cv2.dilate(thresh, kernel_dilate, iterations=1)
    dilate = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_dilate, iterations=2)
    display_image(dilate)
    horizontal_inv = cv2.bitwise_not(dilate)
    display_image(horizontal_inv)
    # masked = cv2.bitwise_and(thresh,image, mask=dilate)
    # display_image(masked)
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 1))
    # kernel_dilate_1 = np.ones((2,4), np.uint8)
    # dilate = cv2.dilate(masked,kernel_dilate_1,iterations=1)
    # display_image(dilate)
    cnts,_ = cv2.findContours(horizontal_inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    sorted_ctrs = sorted(cnts, key=lambda ctr: cv2.boundingRect(ctr)[0])
    for ct in sorted_ctrs:
        # print(ct)
        (x, y, w, h) = cv2.boundingRect(ct)
        roi = horizontal_inv[y:y + h, x:x + w]
        rect = cv2.rectangle(horizontal_inv, (x, y), (x + w, y + h), (0, 255, 0), 1)
        cv2.imshow("",rect)
    cv2.waitKey(0)




bitwise_filter(image_url)
# contour_filter(image_url)
# filter_noise(image_url)
# process_image(image_url)