#!/usr/bin/python
# -*- coding: UTF-8 -*-
import time
from box_overlap import bb_overlap
class Point(object):
    x = 0
    y = 0

    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

if __name__ == '__main__':

    p1 = Point(837,449)
    p2 = Point(1293,609)
    p3 = Point(845,417)
    p4 = Point(1143,565)
    t1 = time.time()
    print(p1.x,p2,p3,p4)
    for i in range (1000):
        print(bb_overlap(p1,p2,p3,p4))
    t2 = time.time()
    print("cost time2:",1000*(t2-t1),"ms")