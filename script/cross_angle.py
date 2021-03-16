# 两条线段的夹角的计算，余弦定理
import math
import numpy as np
import time


# 得到向量的坐标以及向量的模
class Point(object):
    x = 0
    y = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y


class Line(object):
    def __init__(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def vector(self):
        c = (self.x1 - self.x2, self.y1 - self.y2)
        return c

    def length(self):
        d = math.sqrt(pow((self.x1 - self.x2), 2) + pow((self.y1 - self.y2), 2))
        return d


def cross_angle(p1, p2, p3, p4):
    '''
    :param p1: 当前帧目标的坐标点
    :param p2: 10帧前目标的坐标点（可调）
    :param p3: 10帧前目标的坐标点，与当前帧两点合成一条线段（可调）
    :param p4: 20帧前目标的坐标点，与5帧前两点合成一条线段（可调）
    :return: angle
    '''
    first_vector = Line(p1.x, p1.y, p2.x, p2.y)
    second_vector = Line(p3.x, p3.y, p4.x, p4.y)

    vector_multiplication = np.dot(first_vector.vector(), second_vector.vector())
    vector_model = first_vector.length()*second_vector.length()
    result = abs(vector_multiplication / vector_model)
    angle = math.acos(result) * 57.3  # (180/3.14),弧度转化为角度

    return angle


if __name__ == '__main__':
    p1 = Point(1225, 288)  # 当前帧目标的坐标点
    p2 = Point(1254, 386)  # 5帧前目标的坐标点（可调）
    p3 = Point(1285, 312)  # 15帧前目标的坐标点，与当前帧两点合成一条线段（可调）
    p4 = Point(1223, 408)
    res = cross_angle(p1, p2, p3, p4)
