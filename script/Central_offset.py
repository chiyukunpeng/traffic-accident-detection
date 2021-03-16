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

#判断每个目标的停留时间,每判断一次消耗1us
def Central_offset(p1,p2,img_width):

    '''
    :param p1: 当前帧轨迹点
    :param p2: 从当前帧开始，倒数第二十帧轨迹点
    :param img_width: 视频图像宽度，在打开视频时就已获取。该参数用于自适应设定位置偏移阈值
    :return: BOOL
    判断当前帧目标中心点与倒数第十帧目标中心点的距离（偏移），根据图像大小自适应设定位置偏移阈值，若位置偏移大于该阈值，
    则认为目标发生了移动，否则判定为目标静止，计数器加一，当计数器大于（视频帧率fps*60），则认为目标异常停留，返回True。
    设定位置偏移阈值是为了弥补检测与跟踪的定位误差
    '''
    lengthx = p1.x - p2.x
    lengthy = p1.y - p2.y
    dis = math.sqrt((lengthx ** 2) + (lengthy ** 2))
    # print(dis)
    dis < 0.035*img_width

    return dis


# if __name__ == '__main__':
#     p1 = Point(30,100)
#     p2 = Point(50,110)
#     p3 = Point(55,115)
#     width = 1920
#     t1 = time.time()
#     for i in range(1,1000):
#         stop(p1, p2, width)
#         #print(speed_jump(p1,p2,p3,fre))
#     t2 = time.time()
#     print ("cost time4:",1000*(t2-t1),"ms")