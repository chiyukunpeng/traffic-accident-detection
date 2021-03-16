#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time


class Point(object):
    x = 0
    y = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


def bb_overlap(point1, point2, point3, point4, a):
    '''
    说明：图像中，从左往右是 x 轴（0~无穷大），从上往下是 y 轴（0~无穷大），从左往右是宽度 w ，从上往下是高度 h
    :param point1.x: 第一个框的左上角 x 坐标
    :param point1.y: 第一个框的左上角 y 坐标
    :param w1: 第一幅图中的检测框的宽度
    :param h1: 第一幅图中的检测框的高度
    :param point2.x: 第一个框的右下角 x 坐标
    :param point3.x: 第二个框的左上角 x 坐标
    :param w2: 第二幅图中的检测框的宽度
    :param h2: 第二幅图中的检测框的高度
    :param a:  重合度阈值
    :return: 两个如果有交集则返回重叠度（相交矩形框的面积与最小矩形框面积之比（超过重合阈值的输出））, 如果没有交集则返回 0
    '''
    h1 = abs(point2.y - point1.y)
    w1 = abs(point2.x - point1.x)
    h2 = abs(point4.y - point3.y)
    w2 = abs(point4.x - point3.x)
    if point1.x > point3.x + w2:
        return 0
    if point1.y > point3.y + h2:
        return 0
    if point1.x + w1 < point3.x:
        return 0
    if point1.y + h1 < point3.y:
        return 0
    colInt = abs(min(point1.x + w1, point3.x + w2) - max(point1.x, point3.x))  # 重叠面积的宽
    rowInt = abs(min(point1.y + h1, point3.y + h2) - max(point1.y, point3.y))  # 重叠面积的高
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    overlap1 = overlap_area / area1
    overlap2 = overlap_area / area2
    if (overlap1 > a) and (overlap1 > overlap2):
        return overlap1
    if (overlap2 > a) and (overlap2 > overlap1):
        return overlap2
    else:
        return 0


if __name__ == '__main__':

    p1 = Point(50, 50)
    p2 = Point(100, 100)
    p3 = Point(50, 50)
    p4 = Point(150, 150)
    t1 = time.time()
    for i in range(1000):
        bb_overlap(p1, p2, p3, p4)
    t2 = time.time()
    print("cost time2:", 1000*(t2-t1), "ms")