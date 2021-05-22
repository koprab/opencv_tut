import glob
import os
import cv2
import imutils
import numpy as np
from imutils import contours

captcha_folder = r'D:\\captchas\\processed_captcha\\'
extracted_folder = r'D:\\captchas\\extracted_chars\\'

captcha_images = glob.glob(os.path.join(captcha_folder, '*'))
counts = {}

print(f'Total no of images : {len(captcha_images)}')
index = 0


def getAvgContour(cont):
    arealist = []
    for c in cont:
        area = cv2.contourArea(c)
        arealist.append(area)
    avg = sum(arealist)/len(arealist)
    print(f'Avg Contour Area : {avg}')
    return avg


def get_auto_thresh_value(img, sigma=None):
    if sigma is None:
        sigma = 0.3
    elif isinstance(sigma, int):
        # print("int")
        sigma = float(sigma / pow(10, len(str(sigma))))
    elif isinstance(sigma, float):
        pass
    # print(sigma)
    median = np.median(img)
    # print(median)
    lower = int(max(0, ((1 - sigma) * median)))
    upper = int(min(255, ((1 + sigma) * median)))
    return lower, upper


def get_single_char(img):
    file_name = os.path.basename(img)
    correct_text = os.path.splitext(file_name)[0]
    print(file_name)
    print(correct_text)
    # load image
    image = cv2.imread(img)
    # convert to gray
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lower_thresh, upper_thresh = get_auto_thresh_value(gray, 3) # image , percent of reduction
    print((lower_thresh,upper_thresh))
    thresh = cv2.threshold(gray, lower_thresh, upper_thresh, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ERODE, (2, 2))
    eroded = cv2.erode(thresh, erode_kernel, iterations=3)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 2))
    dilated = cv2.dilate(eroded, dilate_kernel, iterations=3)
    # opening = cv2.morphologyEx(dilated, cv2.MORPH_ERODE, dilate_kernel,iterations=2)
    # cv2.imshow("thesh", eroded)
    invert = cv2.bitwise_not(dilated)
    # cv2.imshow("",invert)
    dilate_kernel_1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    dilate = cv2.dilate(invert, dilate_kernel_1, iterations=2)

    cont, hir = cv2.findContours(eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(len(cont))
    avg_area = getAvgContour(cont)
    (sorted_cont, boundingBox) = contours.sort_contours(cont, 'left-to-right')
    print(boundingBox)
    letter_image_regions = []
    if len(cont) >= 4:
        for c in sorted_cont:
            (x, y, w, h) = cv2.boundingRect(c)
            if cv2.contourArea(c) > avg_area:
                # roi = invert[y:y + h, x:x + w]
                if w / h > 1.25:
                    half_width = int(w / 2)
                    letter_image_regions.append((x, y, half_width, h)) ## x ,y, w,h 
                    letter_image_regions.append((x + half_width, y, half_width, h)) ## x ,y ,w, h
                    # roi = invert[y-2:y + half_width, x+half_width:w]
                # roi = invert[y - 2:y + h + 2, x - 2:x + w + 2]
                #     rect = cv2.rectangle(image, (start_x, start_y), (end_x, end_y), (0, 0, 255), 1)
                #     cv2.imshow("", rect)
                else:
                    letter_image_regions.append((x,y,w,h))
                    # roi = invert[y-2 : y+h+2, x-2:x+w+2]
                #     rect = cv2.rectangle(image,(x,y),(x+w, y+h),(0,255,0),1)
                #     cv2.imshow("", rect)
                # cv2.waitKey(0)
            if len(letter_image_regions) != 4:
                continue
    
    for i in letter_image_regions:
        print(i)
        (x, y, w, h) = i
        cv2.rectangle(image,(x-2,y-2),(x+w, y+h),(0,255,0),1)
        cv2.imshow("",image)
        cv2.waitKey(0)
        
    # cv2.imshow("canny", cv2.Canny(invert,100,200,apertureSize=7,L2gradient = True))
    cv2.imshow("",dilate)
    cv2.imshow("or", image)
    cv2.waitKey(0)



get_single_char(captcha_images[20])