import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
class Point(object):
    x = 0
    y = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y
def region_of_interest1(img, rectangle1,rectangle2):
    """
    图像蒙版.
    """
    cv2.rectangle(img, (rectangle1.x, rectangle1.y+40), (rectangle2.x, rectangle2.y+40), (255, 155, 203), -1)
    return img

def region_of_interest2(img,circle):
    """
    图像蒙版.
    """
    cv2.circle(img, (circle.x, circle.y), 20, (104, 231, 250), -1)
    return img


if __name__ == '__main__':
    img = np.zeros((1920, 1080, 3), np.uint8)
    img.fill(255)
    circle = Point(447, 63)
    rectangle1 = Point(550, 550)
    rectangle2 = Point(990, 990)
    region_of_interest1(img, rectangle1, rectangle2)
    winname = 'Aerial view'
    cv2.namedWindow(winname)
    cv2.imshow(winname, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()