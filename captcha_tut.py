import cv2
import numpy as np
import matplotlib.pyplot as plt
image_path = "F:\\processed_captcha\\5o3m.jpg"

def getAvgContour(cont):
    arealist = []
    for c in cont:
        area = cv2.contourArea(c)
        arealist.append(area)
    avg = sum(arealist)/len(arealist)
    # print(f'Avg Contour Area : {avg}')
    return avg


def test(img):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plt.imshow(gray)
    # plt.show()
    y = 25
    x = 10
    h = 60
    w = 195
    # img = np.zeros((height,width), np.uint8)
    thesh = cv2.threshold(gray, 170,255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    img = image.copy()
    image[y:y+h, x:x+w]=[255,255,255]
    # im[y:y+h, x:x+w] = [255,255,255]
    mask = np.zeros(image.shape[:2], dtype="uint8")
    cv2.rectangle(mask,(x,y),(x+w,y+h),(255,255,255),-1)
    i = cv2.bitwise_and(thesh,thesh,mask=mask)
    # print(im.shape)
    # cv2.imshow("im", im)
    # cv2.imshow("",gray)
    n = cv2.bitwise_not(i)
    kernel = np.ones((3,2),np.uint8)
    eroded = cv2.dilate(n,kernel,iterations=2)
    cont,h = cv2.findContours(eroded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    print(len(cont))
    avg_area = getAvgContour(cont)
    for c in cont:
        (x, y, w, h) = cv2.boundingRect(c)
        # if cv2.contourArea(c) > avg_area:
        rect = cv2.rectangle(gray,(x,y),(x+w, y+h),(0,255,0),2)
    cv2.imshow("re",rect)
    # cv2.drawContours(image,cont, -1, (0,255,0),3)
    cv2.imshow("",eroded)
    # cv2.imshow("",i)
    # cv2.imshow("",mask)
    # plt.show()
    cv2.imshow("sd",image)
    cv2.waitKey(0)


test(image_path)

