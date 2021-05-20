import cv2
import matplotlib.pyplot as plt
import imutils
import os

import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"
image_url = r"D:\captchas\processed_captcha\e9ac.jpg"
# image_url = r"D:\captchas\solving_captchas_code_examples\generated_captcha_images\2BTE.png"
OUTPUT_FOLDER = "extracted_letter_images"
import numpy as np
counts = {}


def display_image(text=None,img=None):
    cv2.imshow(text, img)
    cv2.waitKey(0)
    # cv2.destroyAllWindows()

def getContours(img):
    image_height,image_width = img.shape
    ctrs, hier = cv2.findContours(img.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    sorted_ctrs = sorted(ctrs, key=lambda ctr: cv2.boundingRect(ctr)[0])
    print(sorted_ctrs)
    for i, ctr in enumerate(sorted_ctrs):
        x, y, w, h = cv2.boundingRect(ctr)

        half_width = w/h
        if w == image_width and h == image_height:
            pass
        else:
            print(x, y, w, h)
            if half_width >1.25:
                half_width = int(w/2)
                (x,y,w,h) = (x, y, half_width, h)
                roi = img[y+2:y + h+2, x+2:x + w+2]
                display_image("",roi)
                (x,y,w,h)=(x + half_width, y, half_width, h)
                roi = img[y + 2:y + h + 2, x + 2:x + w+2 ]
                display_image("",roi)
            else:
            # [y - 2: y + h + 2, x - 2: x + w + 2]
            # plt.imshow(roi)
            # plt.show()
                roi = img[y + 2:y + h + 2, x + 2:x + w+2]
                display_image("",roi)
        area = w * h
        print(area)
        # if 1000 < area < 2000:
        #     rect = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #     cv2.imshow('rect', rect)
        #
        # cv2.waitKey(0)


def isolated_pixels(image):
    input_image = cv2.imread(image,0)

    ret, thresh_img = cv2.threshold(input_image, 253, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph_img = cv2.morphologyEx(thresh_img, cv2.MORPH_CLOSE, kernel)
    display_image("", morph_img)
    kernel = np.ones((3, 4), np.uint8)
    dilation = cv2.dilate(morph_img, kernel)
    display_image("",dilation)
    # blur = cv2.medianBlur(dilation, 3)
    # display_image("Blur Median", blur)
    blur_1 = cv2.bilateralFilter(dilation, 5, 10, 50)
    display_image("",blur_1)

    # input_image = cv2.threshold(input_image, 254, 255, cv2.THRESH_BINARY)[1]
    # input_image_comp = cv2.bitwise_not(input_image)  # could just use 255-img
    #
    # kernel1 = np.array([[0, 0, 0],
    #                     [0, 1, 0],
    #                     [0, 0, 0]], np.uint8)
    # kernel2 = np.array([[1, 1, 1],
    #                     [1, 0, 1],
    #                     [1, 1, 1]], np.uint8)
    #
    # hitormiss1 = cv2.morphologyEx(input_image, cv2.MORPH_ERODE, kernel1)
    # hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
    # hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)
    #
    # cv2.imshow('isolated.png', hitormiss)
    # cv2.waitKey()
    #
    # bit_not = cv2.bitwise_not(hitormiss)
    # display_image("nit_not",bit_not)
    # del_isolated = cv2.bitwise_and(input_image, input_image, mask=bit_not)
    # cv2.imshow('removed.png', del_isolated)
    # cv2.waitKey()
    # gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
    # display_image("gray",gray)
    # thresh = cv2.adaptiveThreshold(gray,253,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,21,2)
    # display_image("adaptive Thresh", thresh)
    # first_mask = cv2.bitwise_not(thresh)
    # display_image("bitwise_not",first_mask)
    # kernel1 = np.array([[0, 0, 0],
    #                     [0, 1, 0],
    #                     [0, 0, 0]], np.uint8)
    # kernel2 = np.array([[1, 1, 1],
    #                     [1, 0, 1],
    #                     [1, 1, 1]], np.uint8)
    # # hitormiss1 = cv2.morphologyEx(input_image, cv2.MORPH_ERODE, kernel1)
    # # display_image("hitormiss_1",hitormiss1)
    # hitormiss2 = cv2.morphologyEx(first_mask, cv2.MORPH_ERODE, kernel2)
    # display_image("hitormiss_2", hitormiss2)
    # hitormiss = cv2.bitwise_and(hitormiss2, hitormiss2,mask=first_mask)
    # display_image("hitormiss1", hitormiss)



def remove_isolated_pixels_new(image):
    input_image = cv2.imread(image)
    gray = cv2.cvtColor(input_image,cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)[1]
    display_image("Thresh_1",thresh)
    input_image_inv = cv2.bitwise_not(thresh)
    invert_mask = cv2.bitwise_and(gray,gray,mask=input_image_inv)
    display_image("Inverted_Mask",invert_mask)
    masked_img_inv = cv2.bitwise_not(invert_mask)
    kernel = np.ones((4, 3), np.uint8)
    dilation = cv2.dilate(masked_img_inv, kernel)
    display_image("Dilation",dilation)
    blur = cv2.medianBlur(dilation, 3)
    display_image("Blur Median",blur)
    blur_1 = cv2.bilateralFilter(blur, 4, 10, 140)
    display_image("Bilateral_blur",blur_1)
    print(pytesseract.image_to_string(blur_1, config='--psm 11'))
    print(pytesseract.image_to_string(blur, config='--psm 11'))
    revert = cv2.threshold(blur_1,253,255,cv2.THRESH_BINARY)[1]
    display_image("Revert_blur",revert)
    # getContours(revert)

def remove_isolated_pixels(img):
    input_image = cv2.imread(img)
    # input_image = cv2.imread('calc.png', 0)
    input_image = cv2.threshold(input_image, 254, 255, cv2.THRESH_BINARY)[1]
    input_image_comp = cv2.bitwise_not(input_image)  # could just use 255-img

    kernel1 = np.array([[0, 0, 0],
                        [0, 1, 0],
                        [0, 0, 0]], np.uint8)
    kernel2 = np.array([[1, 1, 1],
                        [1, 0, 1],
                        [1, 1, 1]], np.uint8)

    hitormiss1 = cv2.morphologyEx(input_image, cv2.MORPH_ERODE, kernel1)
    hitormiss2 = cv2.morphologyEx(input_image_comp, cv2.MORPH_ERODE, kernel2)
    display_image(hitormiss2)
    display_image(hitormiss1)
    hitormiss = cv2.bitwise_and(hitormiss1, hitormiss2)
    display_image(hitormiss)
    #
    # cv2.imshow('isolated.png', hitormiss)
    # cv2.waitKey()
    hitormiss_comp = cv2.bitwise_not(hitormiss2)  # could just use 255-img
    display_image(hitormiss_comp)
    # del_isolated = cv2.bitwise_and(input_image, input_image, mask=hitormiss2)
    # cv2.imshow('removed.png', del_isolated)
    # cv2.waitKey()
    # display_image(hitormiss_comp)
    # ret, thresh2 = cv2.threshold(hitormiss_comp, 253, 255, cv2.THRESH_BINARY_INV)
    # display_image(thresh2)
    # blur = cv2.medianBlur(hitormiss_comp,5)
    # blur = cv2.blur

def process_captcha(image_path):
    image = cv2.imread(image_path)
    correct_text = os.path.splitext(os.path.basename(image_path))[0]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # hei, thresh = cv2.threshold(gray,0,255, cv2.THRESH_BINARY_INV | cv2.THRESH_BINARY)
    # edge detection
    # edges = cv2.Canny(blur, 100, 200)
    # cv2.floodFill(edges)
    horizontal_inv = cv2.bitwise_not(gray)
    masked_img = cv2.bitwise_and(gray, gray, mask=horizontal_inv)
    # reverse the image back to normal
    masked_img_inv = cv2.bitwise_not(masked_img)

    kernel = np.ones((4, 3), np.uint8)
    dilation = cv2.dilate(masked_img_inv, kernel)
    # blur = cv2.medianBlur(dilation, 3)
    ret, thresh2 = cv2.threshold(dilation, 253, 255, cv2.THRESH_BINARY_INV)
    # th3 = cv2.adaptiveThreshold(dilation, 253, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 11, 2)
    thresh2 = cv2.bitwise_not(thresh2)
    # thresh2 = cv2.bitwise_not(th3)
    # cv2.imshow("masked img", masked_img_inv)
    # cv2.imwrite("result2.jpg", thresh2)
    # print(pytesseract.image_to_string(thresh2))
    # edges = cv2.Canny(thresh2, 10, 200)
    blur = cv2.medianBlur(thresh2,3)
    # blur = cv2.GaussianBlur(th3, (5, 3), 0)
    # blur = cv2.bilateralFilter(thresh2, 20, 110, 150)
    # blur = cv2.medianBlur(th3, 3)
    # edges = cv2.Canny(blur, 100, 200)
    # contours = cv2.findContours(thresh2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    #
    # # Hack for compatibility with different OpenCV versions
    # # contours = contours[0] if imutils.is_cv2() else contours[1]
    # contours = imutils.grab_contours(contours)
    # print(contours)
    display_image("",blur)
    # isolated_pixels(blur)

    print(pytesseract.image_to_string(blur, config='--psm 11'))
    getContours(blur)
    # contours = cv2.findContours(blur.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # #
    # # # Hack for compatibility with different OpenCV versions
    # contours = contours[1] if imutils.is_cv2() else contours[0]
    # # contours = imutils.grab_contours(contours)
    # print(contours)
    #
    # letter_image_regions = []
    #
    # # Now we can loop through each of the four contours and extract the letter
    # # inside of each one
    # for contour in contours:
    #     # Get the rectangle that contains the contour
    #     (x, y, w, h) = cv2.boundingRect(contour)
    #     print((x, y, w, h))
    #     # Compare the width and height of the contour to detect letters that
    #     # are conjoined into one chunk
    #     if w / h > 1.25:
    #         # This contour is too wide to be a single letter!
    #         # Split it in half into two letter regions!
    #         half_width = int(w / 2)
    #         letter_image_regions.append((x, y, half_width, h))
    #         letter_image_regions.append((x + half_width, y, half_width, h))
    #     else:
    #         # This is a normal letter by itself
    #         letter_image_regions.append((x, y, w, h))
    #
    #     # If we found more or less than 4 letters in the captcha, our letter extraction
    #     # didn't work correcly. Skip the image instead of saving bad training data!
    #     # if len(letter_image_regions) != 4:
    #     #     continue
    #
    # letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])
    #
    # print(letter_image_regions)
    # # contours = contours.tolist()
    # for letter_bounding_box, letter_text in zip(letter_image_regions, correct_text):
    #     # Grab the coordinates of the letter in the image
    #     x, y, w, h = letter_bounding_box
    #
    #     # Extract the letter from the original image with a 2-pixel margin around the edge
    #     letter_image = blur[y - 2:y + h + 2, x - 2:x + w + 2]
    #
    #     # Get the folder to save the image in
    #     # save_path = os.path.join(OUTPUT_FOLDER, letter_text)
    #
    #     # if the output directory does not exist, create it
    #     # if not os.path.exists(save_path):
    #     #     os.makedirs(save_path)
    #
    #     # write the letter image to a file
    #     count = counts.get(letter_text, 1)
    #     # p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
    #     # cv2.imwrite(p, letter_image)
    #     display_image(letter_image)
    #     # increment the count for the current key
    #     counts[letter_text] = count + 1


process_captcha(image_url)
# # remove_isolated_pixels(image_url)
# remove_isolated_pixels_new(image_url)
# isolated_pixels(image_url)