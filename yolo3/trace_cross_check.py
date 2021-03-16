# -*- coding: utf-8 -*-
from ctypes import *
import numpy as np
import time
import math

class Point(object):
    x = 0
    y = 0
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

# 计算两条轨迹的交点
def crossCal(line1, line2, line3, line4):
    '''
    :param line1: 第一个目标的当前帧轨迹点
    :param line2: 第一个目标的倒数第30帧轨迹点
    :param line3: 第二个目标的当前帧轨迹点
    :param line4: 第二个目标的倒数第30帧轨迹点
    :return: BOOL
    计算两条轨迹的交点，轨迹线段一般不相交，改为判断两条线段所在直线是否相交
    取每个目标轨迹的最后一帧与倒数第30帧两个点做直线
    '''
    x_denominator = line4.x * line2.y - line4.x * line1.y - line3.x * line2.y + line3.x * line1.y - line2.x * line4.y + line2.x * line3.y + line1.x * line4.y - line1.x * line3.y;
    x_member = line3.y * line4.x * line2.x - line4.y * line3.x * line2.x - line3.y * line4.x * line1.x + line4.y * line3.x * line1.x - line1.y * line2.x * line4.x + line2.y * line1.x * line4.x + line1.y * line2.x * line3.x - line2.y * line1.x * line3.x;
    cross_point = Point(x=0, y=0)
    if (x_denominator == 0):
        cross_point.x = 0;
    else:
        cross_point.x = x_member / x_denominator;
        # 限制横坐标x的范围在线段内，否则就是直线相交的判断了
    if (cross_point.x < line3.x or cross_point.x > line4.x):
        cross_point.x = 0

    y_denominator = line4.y * line2.x - line4.y * line1.x - line3.y * line2.x + line1.x * line3.y - line2.y * line4.x + line2.y * line3.x + line1.y * line4.x - line1.y * line3.x;
    y_member = -line3.y * line4.x * line2.y + line4.y * line3.x * line2.y + line3.y * line4.x * line1.y - line4.y * line3.x * line1.y + line1.y * line2.x * line4.y - line1.y * line2.x * line3.y - line2.y * line1.x * line4.y + line2.y * line1.x * line3.y;

    if (y_denominator == 0):
        cross_point.y = 0;
    else:
        cross_point.y = y_member / y_denominator;
    # print(cross_point.x,cross_point.y)
    return int(cross_point.x), int(cross_point.y)

if __name__ == '__main__':
    p1 = Point(30,100)
    p2 = Point(50,110)
    p3 = Point(55,115)
    p4 = Point(60,115)
    fre = 0.04
    t1 = time.time()
    for i in range(1000):
        crossCal(p1, p2, p3, p4)
        #print(speed_jump(p1,p2,p3,fre))
    t2 = time.time()
    print ("cost time:",1000*(t2-t1),"ms")