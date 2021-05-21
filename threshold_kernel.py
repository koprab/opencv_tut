import cv2
import imutils
import numpy as np
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract"

image_url = r'captcha_images\bn6Q.jpg'


def getAvgContour(cont):
    arealist = []
    for c in cont:
        area = cv2.contourArea(c)
        arealist.append(area)
    avg = sum(arealist)/len(arealist)
    print(f'Avg Contour Area : {avg}')
    return avg


def combine_horizontally(image_names,padding=20):
    images = []
    max_height = 0  # find the max height of all the images
    total_width = 0  # the total width of the images (horizontal stacking)
    for name in image_names:
        # open all images and find their sizes
        # img = cv2.imread(name)
        images.append(name)
        image_height = name.shape[0]
        image_width = name.shape[1]
        if image_height > max_height:
            max_height = image_height
        # add all the images widths
        total_width += image_width
    # create a new array with a size large enough to contain all the images
    # also add padding size for all the images except the last one
    final_image = np.zeros((max_height, (len(image_names)-1)*padding+ total_width), dtype=np.uint8)
    current_x = 0  # keep track of where your current image was last placed in the x coordinate
    for image in images:
        # add an image to the final array and increment the x coordinate
        height = image.shape[0]
        width = image.shape[1]
        final_image[:height,current_x :width+current_x] = image
        #add the padding between the images
        current_x += width+padding
    return final_image


def get_contour_precedence(contour, cols):
    origin = cv2.boundingRect(contour)
    return origin[1] * cols + origin[0]


def sort_contours(cnts, method="left-to-right"):
    # initialize the reverse flag and sort index
    reverse = False
    i = 0
    # handle if we need to sort in reverse
    if method == "right-to-left" or method == "bottom-to-top":
        reverse = True
    # handle if we are sorting against the y-coordinate rather than
    # the x-coordinate of the bounding box
    if method == "top-to-bottom" or method == "bottom-to-top":
        i = 1
    # construct the list of bounding boxes and sort them from top to
    # bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
        key=lambda b:b[1][i], reverse=reverse))
    # return the list of sorted contours and bounding boxes
    return (cnts, boundingBoxes)


def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    # return cv2.hconcat(im_list_resize)
    return cv2.hconcat(im_list_resize)


def appy_threshold(img):
    img_o = cv2.imread(img)
    image = cv2.imread(img, 0)

    sigma = 0.3
    median = np.median(image)
    lower = int(max(0, ((1-sigma) * median)))
    upper = int(min(255,((1+sigma)*median)))
    print(f'median of image :{median}')
    print(lower)
    print(upper)
    print(image.shape)
    thresh = cv2.threshold(image,150,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    thresh_auto = cv2.threshold(image,lower,upper,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # erode
    # for i in range(2,5):
    #     for j in range(2,5):
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    eroded = cv2.erode(thresh_auto, erode_kernel, iterations=2)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 2))
    dilated = cv2.dilate(eroded, dilate_kernel, iterations=3)

    cont,hir = cv2.findContours(eroded,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # print(cont)
    print(len(cont))
    # print(f'Avg of contours area : {sum(cont)}')
    avgArea = getAvgContour(cont)
    # cont = sorted(cont,key=cv2.contourArea, reverse=True)
    # cont.sort(key=lambda x:get_contour_precedence(x, eroded.shape[1]))
    # sorted_ctrs = sorted(cont, key=lambda ctr: cv2.boundingRect(ctr)[0] + cv2.boundingRect(ctr)[1] * eroded.shape[1])
    sorted_ctrs, bounding = sort_contours(cont,'left-to-right')
    invert = cv2.bitwise_not(eroded)
    # cv2.drawContours(image,cont,-1,(255,0,255),3)
    # cn = imutils.grab_contours(cont)
    adptive_thresh = cv2.adaptiveThreshold(invert,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,2)
    cv2.imshow("ad",adptive_thresh)
    cv2.waitKey(0)
    print(pytesseract.image_to_string(invert,config='--psm 7 --oem 2'))
    cont_list = []
    for c in sorted_ctrs:
        (x,y,w,h) = cv2.boundingRect(c)
        if cv2.contourArea(c) > avgArea:
            # roi = invert[y:y + h, x:x + w]
            roi = invert[y-2:y + h+2, x-2:x + w+2]
            rect = cv2.rectangle(img_o, (x, y), (x + w, y + h), (0,0, 255), 1)
            cv2.imshow("",rect)
            cv2.waitKey(0)
            cont_list.append(roi)

            # print(f"ROI TESerract - {pytesseract.image_to_string(roi,config='--psm 11')}")
        # cv2.imwrite(f"x_{x}.png", roi)
    # cv2.imshow(f"i={i}:j={j}", eroded)
    # print(cont_list)
    im_h_resize = hconcat_resize_min(cont_list)
    # im_h_resize = combine_horizontally(cont_list,2)
    new_im = cv2.copyMakeBorder(im_h_resize, 20, 20, 30, 30, cv2.BORDER_CONSTANT,value=[255,255,255])
    adptive_thresh_new = cv2.adaptiveThreshold(new_im, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 2)
    print(f'Text from OCR after joining => {pytesseract.image_to_string(adptive_thresh_new,config="--psm 11 --oem 2")}')
    dilate_kernel_new = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 2))
    # dilate_after = cv2.morphologyEx(new_im,cv2.MORPH_ERODE,dilate_kernel_new,iterations=1)
    # cv2.imshow("d_a", dilate_after)
    # cv2.imshow("combined", im_h_resize)
    cv2.imshow("re_sized",adptive_thresh_new)
    # cv2.waitKey(0)
    # um = unsharp_mask(invert)
    # cv2.imshow("original", image)

    cv2.imshow("original", image)
    cv2.imshow("Thresh", thresh)
    cv2.imshow("auto", thresh_auto)
    cv2.imshow("Eroded", eroded)
    cv2.imshow("dilated", dilated)
    cv2.imshow("invert", invert)
    cv2.imshow("colored", img_o)



    cv2.waitKey(0)

appy_threshold(image_url)