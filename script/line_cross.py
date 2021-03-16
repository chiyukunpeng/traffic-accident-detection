#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
import math
import cmath
import sys
global D
class Point(object):
    x = 0
    y = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

def cross(p1, p2, p3):  # 叉积判定
    x1 = p2.x - p1.x
    y1 = p2.y - p1.y
    x2 = p3.x - p1.x
    y2 = p3.y - p1.y
    return x1 * y2 - x2 * y1

# def line_extend1(p1,p2,w,h):  # 据起始点和终止点，延长线段，70为延长横坐标70个像素点，为阈值可调
#     k = (-p1.y + p2.y) / (p1.x - p2.x+0.000001)  # 因为坐标轴以左上角为起点
#     b = (-p2.y) - k * p2.x
#     if k < 0:
#         x1 = p1.x + 100
#         y1 = -(k * x1 + b)
#         pe = Point(x1,y1)
#         return pe
#     if k > 0:
#         x1 = p1.x - 100
#         y1 = -(k * x1 + b)
#         pe = Point(x1,y1)
#         return pe

def line_extend(p1,p2,R):  # 据起始点和终止点，延长线段，70为延长横坐标70个像素点，为阈值可调
    k = (-p1.y + p2.y) / (p1.x - p2.x+0.000001)  # 因为坐标轴以左上角为起点
    b = (-p2.y) - k * p2.x
    a = 1 + k*k
    b1 = -2 * p1.x + 2 * k * b + 2 * k * p1.y
    c = p1.x * p1.x + p1.y * p1.y + b * b + 2 * p1.y * b - R * R
    discriminant = (b1 * b1) - (4 * a * c)

    if discriminant == 0:
        x1 = -(b / (2 * a))
    else:
        if discriminant > 0:
            root = math.sqrt(discriminant)
        else:   #discriminant < 0
            root = cmath.sqrt(discriminant)
        x1 = (-b1 + root) / (2 * a)
        x2 = (-b1 - root) / (2 * a)

    # x1 = (-b1 + root) / (2 * a)
    # x2 = (-b1 - root) / (2 * a)

    if p1.x > p2.x:
        if discriminant == 0:
            if p1.y < p2.y:
                y1 = p1.y - math.sqrt(R * R - (x1 - p1.x) * (x1 - p1.x))
                pe = Point(x1, y1)
            else :
                y1 = math.sqrt(R * R - (x1 - p1.x) * (x1 - p1.x)) + p1.y
                pe = Point(x1, y1)
        else :
            if p1.y < p2.y:
                y1 = p1.y - math.sqrt(R * R - (x1 - p1.x) * (x1 - p1.x))
                pe = Point(x1, y1)
            else :
                y1 = math.sqrt(R * R - (x1 - p1.x) * (x1 - p1.x)) + p1.y
                pe = Point(x1, y1)
        return pe
    if p1.x < p2.x:
        if discriminant == 0:
            if p1.y < p2.y:
                y1 = p1.y - math.sqrt(R * R - (x1 - p1.x) * (x1 - p1.x))
                pe = Point(x1, y1)
            else :
                y1 = math.sqrt(R * R - (x1 - p1.x) * (x1 - p1.x)) + p1.y
                pe = Point(x1, y1)
        else:
            if p1.y < p2.y:
                y1 = p1.y - math.sqrt(R * R - (x2 - p1.x) * (x2 - p1.x))
                pe = Point(x2, y1)
            else :
                y1 = math.sqrt(R * R - (x2 - p1.x) * (x2 - p1.x)) + p1.y
                pe = Point(x2, y1)
        return pe

def line_cross(p1, p2, p3, p4, w, h ,R):  # 判断两线段是否相交
    """
    :param p1: 第一条线段的起始点
    :param p2: 第一条线段10帧之前的点
    :param w:  宽的分辨率
    :param h:  高的分辨率
    :param R:  绝对延伸距离
    :return: bool，是否相交
    """
    pe1 = line_extend(p1, p2 ,R)#第一条线段延长点
    pe2 = line_extend(p3, p4 ,R)#第二条线段延长点
    # 矩形判定，以l1、l2为对角线的矩形必相交，否则两线段不相交
    if not (pe1 is None or pe2 is None or p2 is None or p4 is None):
        if (max(pe1.x, p2.x) >= min(pe2.x, p4.x)  # 矩形1最右端大于矩形2最左端
        #将两条线段表现成两个矩形框，初始判断是否相交
                and max(pe2.x, p4.x) >= min(pe1.x, p2.x)  # 矩形2最右端大于矩形1最左端
                and max(pe1.y, p2.y) >= min(pe2.y, p4.y)  # 矩形1最高端大于矩形2最低端
                and max(pe2.y, p4.y) >= min(pe1.y, p2.y)  # 矩形2最高端大于矩形1最低端
                and (pe1.x < w and pe1.y < h and pe1.x > 0 and pe1.y > 0 )
                and (pe2.x < w and pe2.y < h and pe2.x > 0 and pe2.y > 0)):
                if (cross(pe1, p2, pe2) * cross(pe1, p2, p4) <= 0  # 利用叉积判断线段是否相交
                        and cross(pe2, p4, pe1) * cross(pe2, p4, p2) <= 0):
                    D = 1
                else:
                    D = 0
        else:
            D = 0
    else:
        D = 0
    return D

# if __name__ == '__main__':
#
#         p1 = Point(526, 528)
#         p2 = Point(459, 510)
#         p3 = Point(635, 458)
#         p4 = Point(649, 489)
#         R = 200
#
# d=line_cross(p1, p2, p3, p4,1920,1080,R)
# print("D=", d)
# d=line_extend(p1,p2,R)
# print("p=", d.x,d.y)
# t1 = time.time()
# for i in range(1000):
#     line_cross(p1, p2, p3, p4, 1920, 1080,R)
# t2 = time.time()
# print("cost time2:", 1000 * (t2 - t1), "ms")
