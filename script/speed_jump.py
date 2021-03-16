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


# 判断速度是否发生突变，每判断一次消耗2us
def speed_jump(c1, c2, c3, inter_time):
    '''
       1、去除每辆车的前10帧，防止刚进入画面检测跳动
       2、计算轨迹倒数第十帧和第二十帧之间的速度 s1
       3、计算轨迹倒数第一帧和第十帧之间的速度 s2
       4、判断这两个速度的变化率
       c1,c2,c3从当前帧计算，倒数第10帧轨迹点，倒数第5帧轨迹点，当前帧轨迹点
       inter_time根据帧率计算而来的两帧间隔时间
    '''
    # math calculation 经测试，math方法时间效率高于numpy的6倍，numpy需要将坐标转换为array
    length1x = c1.x - c2.x
    length1y = c1.y - c2.y
    length2x = c2.x - c3.x
    length2y = c2.y - c3.y
    dis1 = math.sqrt((length1x ** 2) + (length1y ** 2))
    dis2 = math.sqrt((length2x ** 2) + (length2y ** 2))

    s1 = dis1 / (inter_time*9)
    s2 = dis2 / (inter_time*9)
    speed_jump = abs(s2 - s1) / (s1 + 0.000001)

    return s2, speed_jump


if __name__ == '__main__':
    p1 = Point(30, 100)
    p2 = Point(50, 110)
    p3 = Point(55, 115)
    fre = 0.04
    speed_jump(p1, p2, p3, fre)



