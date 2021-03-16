from __future__ import division, print_function, absolute_import
import warnings
import cv2
import csv
import os
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

warnings.filterwarnings('ignore')
import tensorflow as tf
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    ...

from keras import backend as K

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)

from box_overlap import bb_overlap
from speed_jump import speed_jump
from Curve_catastrophe import cross_angle
from line_cross import line_cross
from Central_offset import Central_offset
from bird_eye import perspective_transform

import sys
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtMultimedia import *
from PyQt5.QtGui import *
from myui import Ui_Form


def on_EVENT_LBUTTONDOWN(event, x, y, flags, param):
    global img
    if event == cv2.EVENT_LBUTTONDOWN:
        xy = "%d,%d" % (x, y)
        coordinate.append(x)
        print("x,y:", xy)
        cv2.circle(img, (x, y), 1, (255, 0, 0), thickness=-1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 0), thickness=2)
        cv2.imshow("image", img)
        cv2.waitKey(5000)
        print('123A')

def id_change(id1, id2):
    global a, b
    if (a == id1 and b == id2) or (b == id1 and a == id2):
         return True
    else:
         a = id1
         b = id2
         return False


class newui(QWidget,Ui_Form):
    def __init__(self,parent = None):
        super(newui,self) .__init__(parent)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.openvideo)
        self.pushButton_2.clicked.connect(self.pause)
        self.pushButton_3.clicked.connect(self.jixu)
        self.pushButton_4.clicked.connect(self.wenjianweizhi)
        self.pushButton_5.clicked.connect(self.shipinweizhi)
        self.listWidget.clicked.connect(self.shipinbofang)

    def pause(self):
        self.player1.pause()
        self.player2.pause()
        self.player3.pause()
        self.player4.pause()

    def jixu(self):
        self.player1.play()
        self.player2.play()
        self.player3.play()
        self.player4.play()

    def shipinweizhi(self):
        path = os.getcwd()
        os.system("start explorer " + path + '\out')

    def wenjianweizhi(self):
        path = os.getcwd()
        os.system("start explorer " + path + '\outfile')
        self.listWidget.clear()
        currentDirPath = path + '\outfile'
        for filename in os.listdir(currentDirPath):
            print(filename)
            self.listWidget.addItem(filename)

    def shipinbofang(self):
        row1 = self.listWidget.currentIndex().row()
        name = self.listWidget.item(row1).text()
        path = os.getcwd()
        pathdir = path + '/outfile/' + name
        self.player1 = QMediaPlayer()
        self.player1.setVideoOutput(self.widget)
        self.player1.setMedia(QMediaContent(QUrl.fromLocalFile(pathdir)))
        self.player1.play()



    def openvideo(self):
        yolo = YOLO()
        str1 = QFileDialog()
        str1.setFileMode(QFileDialog.AnyFile)

        if str1.exec_():
            strname = str1.selectedFiles()
            path = os.getcwd()
            strname2 = path + '/outfile/out3.mkv'
            strname3 = path + '/outfile/out3_flow.mkv'
            strname4 = path + '/outfile/out3_text.mkv'
            strname5 = path + '/outfile/out3_bird.mkv'

            print(strname[0])
            video_name = 'out'
            global inc_s1,inc_s2,inc_sj1,inc_sj2,inc_a1,inc_a2,inc_co1,inc_co2,inc_point,inc_id1,inc_id2,inc_time
            inc_s1=inc_s2=inc_sj1=inc_sj2=inc_a1=inc_a2=inc_co1=inc_co2=inc_point=inc_id1=inc_id2=inc_time=0
            global i, m, w, h, pix2meter, id1, id2, img
            id1 = 0
            id2 = 0
            # 定义车辆信息存储字典
            car_dic = {}

            # 参数阈值
            max_cosine_distance = 0.3
            nn_budget = None
            nms_max_overlap = 1.0

            # deep_sort
            model_filename = 'model_data/mars-small128.pb'
            encoder = gdet.create_box_encoder(model_filename, batch_size=1)
            metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
            tracker = Tracker(metric, 60)

            writeVideo_flag = True

            # 读入本地视频
            video_capture = cv2.VideoCapture(strname[0])

            # 创建数据文件
            csvfile = open("out/"+"incident" + "_" + "information" + "_" + video_name + '.csv', "w", newline="")
            csvwriter = csv.writer(csvfile, dialect='excel')
            csvwriter.writerow(
               ["事故帧号", "事故发生时间", "事故方ID", "事故发生点", "速度1(km/h)", "速度2(km/h)", "速度变化率1", "速度变化率2", "转角1", "转角2",
                "中心偏移量1", "中心偏移量2", "是否误检"])

            # 单位像素与米的换算关系
            reply = QMessageBox.information(self,"window","Please choose whether to display the speed parameter or not.",QMessageBox.Yes | QMessageBox.No)
            if reply == QMessageBox.Yes:
                c = 1
            else:
                c = 0
            if c == 1:
                video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, img = video_capture.read()
                print("请左键点击图中两点")
                cv2.namedWindow("image")
                cv2.setMouseCallback("image", on_EVENT_LBUTTONDOWN)
                cv2.imshow("image", img)
                cv2.waitKey(0)
                x1 = int(coordinate[0])
                x2 = int(coordinate[1])
                text,ok = QInputDialog.getText(self,'window','输入两点实际距离(m)')
                if ok:
                    dis = float(text)
                if not ok:
                    text, ok = QInputDialog.getText(self, 'window', '输入两点实际距离(m)')
                    if ok:
                        dis = float(text)
                #dis_str = input('请输入这两点x方向实际距离(单位:m),按回车结束：')
                if dis <= 0 or dis > 50:
                    text, ok = QInputDialog.getText(self, 'window', '范围超出,重新输入两点实际距离(m)')
                    if ok:
                        dis = float(text)
                    if not ok:
                        text, ok = QInputDialog.getText(self, 'window', '范围超出,重新输入两点实际距离(m)')
                        if ok:
                            dis = float(text)

                pix2meter = dis / abs(x1 - x2)
            else:
                pix2meter = 0

            # 写入视频
            if writeVideo_flag:
                w = int(video_capture.get(3))
                h = int(video_capture.get(4))
                fps = int(video_capture.get(5))
                print("fps:",fps)
                frequency = 1 / fps
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out1 = cv2.VideoWriter(strname2, fourcc, fps, (w, h))
                out2 = cv2.VideoWriter(strname3, fourcc, fps, (w, h))
                out3 = cv2.VideoWriter(strname4, fourcc, fps, (w, h))
                out4 = cv2.VideoWriter(strname5, fourcc, fps, (w, h))
                frame_index = -1
                # 事故抓拍图片
                img_dir = path + '/out'
                #img_dir = "E:/my_program/事故检测/道路监控事故视频/out"

            # 轨迹颜色
            track_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
                            (0, 255, 255), (255, 0, 255), (255, 127, 255),
                            (127, 0, 255), (127, 0, 127)]

            ret, frame1 = video_capture.read()
            # 前一帧
            prev = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255
            mynum = 0
            self.tableWidget.setRowCount(2)
            self.tableWidget.setColumnCount(7)
            self.tableWidget.setHorizontalHeaderLabels(['事故地点', '时间', 'ID号', '速度', '速度变化率', '转角', '中心偏移量'])

            while video_capture.isOpened():
                ret, frame = video_capture.read()
                if ret == True:
                    # t1 = time.time()
                    warped = perspective_transform(frame)
                    # 后一帧
                    next = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    # 稠密光流跟踪（0.5为金字塔放缩比率，3为金字塔层级数目，15为窗口大小，3为迭代次数，5为多项式最高阶，1.2为高斯系数）
                    flow = cv2.calcOpticalFlowFarneback(prev, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    # 坐标系转换
                    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                    hsv[..., 0] = ang * 180 / np.pi / 2
                    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

                    # BGR to RGB
                    image = Image.fromarray(frame[..., ::-1])
                    # 车辆检测
                    boxs = yolo.detect_image(image)
                    # 特征提取
                    features = encoder(frame, boxs)
                    detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]

                    boxes = np.array([d.tlwh for d in detections])
                    scores = np.array([d.confidence for d in detections])
                    indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
                    # 最终的检测结果
                    detections = [detections[i] for i in indices]

                    k = 0
                    image1 = np.zeros((1080, 1920, 3), np.uint8)
                    image1.fill(255)

                    # tracker
                    tracker.predict()
                    tracker.update(detections)
                    for track in tracker.tracks:
                        if not track.is_confirmed() or track.time_since_update > 1:
                            continue
                        bbox = track.to_tlbr()
                        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0), 2)
                        cv2.putText(frame, str(track.track_id), (int(bbox[0]), int(bbox[1])), 0, 5e-3 * 200,
                                    (0, 255, 0), 2)

                        # 0-中心点x,1-中心点y，2-宽度，3-高度
                        center = [int((bbox[0] + bbox[2]) / 2), int((bbox[1] + bbox[3]) / 2), int(bbox[2] - bbox[0]),
                                  int(bbox[3] - bbox[1])]
                        # 如果没有当前车的索引，则创建
                        if not "%d" % track.track_id in car_dic:
                            # 创建当前id的字典：{轨迹，车牌，丢帧计数器，泊车计数器}
                            car_dic["%d" % track.track_id] = {"trace": [], "plate": "沪A3   ", "traced_frames": 40,
                                                              "parked": 0}
                            car_dic["%d" % track.track_id]["trace"].append(center)
                            car_dic["%d" % track.track_id]["traced_frames"] += 1
                        # 如果有，直接写入
                        else:
                            car_dic["%d" % track.track_id]["trace"].append(center)
                            car_dic["%d" % track.track_id]["traced_frames"] += 1

                        # for o in car_dic:
                        p = int(track.track_id)
                        # 写入内容格式为car_id：[track_id],speed:[num],coordinate:(center point),box_width:[width],box_height:[height]
                        cnt = "object: %.2d " % p + "," + "    coordinate:  (%.4d,%.4d)" % (
                        bbox[0], bbox[1]) + "," + "    box_width:  %.4d" % (
                                          bbox[2] - bbox[0]) + "," + "    box_height:  %.4d" % (bbox[3] - bbox[1])
                        cv2.putText(image1, cnt, (10, 60 + 60 * k), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 0), 1,
                                    cv2.LINE_AA)
                        k += 1

                    # 遍历字典的key值
                    for s in car_dic:
                        i = int(s)
                        xlist, ylist, wlist, hlist = [], [], [], []
                        # bbox左上角坐标
                        bbox_x1 = int(car_dic["%d" % i]["trace"][-1][0]) - int(car_dic["%d" % i]["trace"][-1][2]) // 2
                        bbox_y1 = int(car_dic["%d" % i]["trace"][-1][1]) - int(car_dic["%d" % i]["trace"][-1][3]) // 2
                        # 限制轨迹最大长度
                        if len(car_dic["%d" % i]["trace"]) > 60:
                            for k in range(len(car_dic["%d" % i]["trace"]) - 60):
                                del car_dic["%d" % i]["trace"][k]

                        if len(car_dic["%d" % i]["trace"]) > 2:
                            # 绘制轨迹
                            for j in range(1, len(car_dic["%d" % i]["trace"]) - 1):
                                pot1_x = car_dic["%d" % i]["trace"][j][0]
                                pot1_y = car_dic["%d" % i]["trace"][j][1]
                                pot2_x = car_dic["%d" % i]["trace"][j + 1][0]
                                pot2_y = car_dic["%d" % i]["trace"][j + 1][1]
                                clr = i % 9
                                cv2.line(frame, (pot1_x, pot1_y), (pot2_x, pot2_y), track_colors[clr], 2)

                                # 一维列表赋值(x,y,w,h)
                                xlist.append(pot1_x)
                                ylist.append(pot1_y)
                                wlist.append(car_dic["%d" % i]["trace"][j][2])
                                hlist.append(car_dic["%d" % i]["trace"][j][3])

                        # 判断条件
                        for t in car_dic:
                            m = int(t)
                            # 排除取到相同类别
                            if m == i:
                                continue

                            # 当每个目标出现超过20帧开始判断
                            if len(car_dic["%d" % i]["trace"]) > 20:
                                c1_x = car_dic["%d" % i]["trace"][-20][0]
                                c1_y = car_dic["%d" % i]["trace"][-20][1]
                                c2_x = car_dic["%d" % i]["trace"][-10][0]
                                c2_y = car_dic["%d" % i]["trace"][-10][1]
                                c3_x = car_dic["%d" % i]["trace"][-5][0]
                                c3_y = car_dic["%d" % i]["trace"][-5][1]
                                c4_x = car_dic["%d" % i]["trace"][-1][0]
                                c4_y = car_dic["%d" % i]["trace"][-1][1]

                                c_point1 = Point(c1_x, c1_y)
                                c_point2 = Point(c2_x, c2_y)
                                c_point3 = Point(c3_x, c3_y)
                                c_point4 = Point(c4_x, c4_y)

                            # 当每个目标出现超过20帧开始判断
                            if len(car_dic["%d" % m]["trace"]) > 20:
                                c5_x = car_dic["%d" % m]["trace"][-20][0]
                                c5_y = car_dic["%d" % m]["trace"][-20][1]
                                c6_x = car_dic["%d" % m]["trace"][-10][0]
                                c6_y = car_dic["%d" % m]["trace"][-10][1]
                                c7_x = car_dic["%d" % m]["trace"][-5][0]
                                c7_y = car_dic["%d" % m]["trace"][-5][1]
                                c8_x = car_dic["%d" % m]["trace"][-1][0]
                                c8_y = car_dic["%d" % m]["trace"][-1][1]

                                c_point5 = Point(c5_x, c5_y)
                                c_point6 = Point(c6_x, c6_y)
                                c_point7 = Point(c7_x, c7_y)
                                c_point8 = Point(c8_x, c8_y)

                            # 计算字典中任取两个目标的bbox框的左上角和右下角，每两个目标一匹配，计算这两个目标之间的重合度
                            bbox1_leftx = abs(
                                int(car_dic["%d" % i]["trace"][-1][0]) - int(car_dic["%d" % i]["trace"][-1][2]) // 2)
                            bbox1_lefty = abs(
                                int(car_dic["%d" % i]["trace"][-1][1]) - int(car_dic["%d" % i]["trace"][-1][3]) // 2)
                            bbox1_leftp = Point(bbox1_leftx, bbox1_lefty)
                            bbox1_rightx = abs(
                                int(car_dic["%d" % i]["trace"][-1][0]) + int(car_dic["%d" % i]["trace"][-1][2]) // 2)
                            bbox1_righty = abs(
                                int(car_dic["%d" % i]["trace"][-1][1]) + int(car_dic["%d" % i]["trace"][-1][3]) // 2)
                            bbox1_rightp = Point(bbox1_rightx, bbox1_righty)
                            bbox2_leftx = abs(
                                int(car_dic["%d" % m]["trace"][-1][0]) - int(car_dic["%d" % m]["trace"][-1][2]) // 2)
                            bbox2_lefty = abs(
                                int(car_dic["%d" % m]["trace"][-1][1]) - int(car_dic["%d" % m]["trace"][-1][3]) // 2)
                            bbox2_leftp = Point(bbox2_leftx, bbox2_lefty)
                            bbox2_rightx = abs(
                                int(car_dic["%d" % m]["trace"][-1][0]) + int(car_dic["%d" % m]["trace"][-1][2]) // 2)
                            bbox2_righty = abs(
                                int(car_dic["%d" % m]["trace"][-1][1]) + int(car_dic["%d" % m]["trace"][-1][3]) // 2)
                            bbox2_rightp = Point(bbox2_rightx, bbox2_righty)
                            # 计算两个框的高宽比
                            h_w1 = (bbox1_righty - bbox1_lefty) / (bbox1_rightx - bbox1_leftx)
                            h_w2 = (bbox2_righty - bbox2_lefty) / (bbox2_rightx - bbox2_leftx)
                            # 排除同时取到两个人的情形，因为默认人与人之间不会发生交通事故
                            if h_w1 > 1.5 and h_w2 > 1.5:
                                continue

                            if len(car_dic["%d" % i]["trace"]) > 20 and len(car_dic["%d" % m]["trace"]) > 20:
                                Central_offset1 = Central_offset(c_point4, c_point1, w)
                                Central_offset2 = Central_offset(c_point8, c_point5, w)
                                dis = 0.02 * w
                                R = 200
                                # 判别条件1：重合度，延长后轨迹相交，轨迹偏移量
                                if bb_overlap(bbox1_leftp, bbox1_rightp, bbox2_leftp,
                                              bbox2_rightp) > 0.2 and line_cross(c_point4, c_point1, c_point8, c_point5,
                                                                                 w, h,
                                                                                 R) and Central_offset1 > dis and Central_offset2 > dis:
                                    print("overlap and line_cross and Central_offset satisfied")

                                    s1, speed_jump1 = speed_jump(c_point1, c_point2, c_point4, frequency)
                                    s2, speed_jump2 = speed_jump(c_point5, c_point6, c_point8, frequency)
                                    angle1 = cross_angle(c_point4, c_point2, c_point2, c_point1)
                                    angle2 = cross_angle(c_point8, c_point6, c_point6, c_point5)

                                    # 判别条件2：速度突变，轨迹是否发生较大转角
                                    if speed_jump1 > 0.5 or angle1 > 25 or speed_jump2 > 0.5 or angle2 > 25:
                                        id1 = i
                                        id2 = m
                                        # 计算事故发生时间
                                        s = frame_index / fps
                                        min = s / 60
                                        hour = s / 3600
                                        video_time = '%.2d' % hour + ':' + '%.2d' % min + ':' + '%.2d' % s
                                        cv2.putText(frame, "incident happened", (bbox_x1, bbox_y1), 0, 5e-3 * 150,
                                                    (0, 0, 255), 2)
                                        cv2.putText(frame, video_time, (50, 50), 0, 5e-3 * 200, (0, 0, 255), 2)
                                        img_name = str(i) + "vs" + str(m)



                                        # 保存事故图片
                                        if not id_change(i, m):
                                            # 事故参数
                                            print("事故发生点:", (c4_x, c4_y))
                                            print("物体%d速度:%.2f (km/h)" % (id1, (s1 * pix2meter * 3.6)))
                                            print("物体%d速度:%.2f (km/h)" % (id2, (s2 * pix2meter * 3.6)))
                                            print("物体%d速度变化率:%.2f" % (id1, speed_jump1))
                                            print("物体%d速度变化率:%.2f" % (id2, speed_jump2))
                                            print("物体%d转角:%.2f" % (id1, angle1))
                                            print("物体%d转角:%.2f" % (id2, angle2))
                                            print("物体%d中心偏移量:%.2f" % (id1, Central_offset1))
                                            print("物体%d中心偏移量:%.2f" % (id2, Central_offset2))
                                            print("事故发生时间:", video_time)
                                            each_video_full_path = os.path.join(img_dir, video_name)
                                            cv2.imwrite(
                                                each_video_full_path + "_" + str(frame_index) + "_" + img_name + ".jpg",
                                                frame)

                                            inc_point = (c4_x, c4_y)
                                            inc_id1 = id1
                                            inc_id2 = id2
                                            inc_a1 = '%.2f' %( angle1)
                                            inc_a2 ='%.2f' %( angle2)
                                            inc_co1 ='%.2f' %( Central_offset1)
                                            inc_co2 = '%.2f' %(Central_offset2)
                                            inc_sj1 = '%.2f' %(speed_jump1)
                                            inc_sj2 = '%.2f' %(speed_jump2)
                                            inc_s1 = '%.2f' %(s1 * pix2meter * 3.6)
                                            inc_s2 ='%.2f' %( s2 * pix2meter * 3.6)
                                            inc_time = video_time

                                            rowcount = self.tableWidget.rowCount();
                                            self.tableWidget.insertRow(rowcount)
                                            rowcount1 = self.tableWidget.rowCount();
                                            self.tableWidget.insertRow(rowcount1)
                                            self.tableWidget.setHorizontalHeaderLabels(
                                                ['事故地点', '时间', 'ID号', '速度', '速度变化率', '转角', '中心偏移量'])
                                            newitem = QTableWidgetItem(str(inc_point))
                                            self.tableWidget.setItem(mynum, 0, newitem)
                                            self.tableWidget.setItem(mynum + 1, 0, newitem)
                                            newitem = QTableWidgetItem(str(inc_time))
                                            self.tableWidget.setItem(mynum, 1, newitem)
                                            self.tableWidget.setItem(mynum + 1, 1, newitem)
                                            newitem = QTableWidgetItem(str(inc_id1))
                                            self.tableWidget.setItem(mynum, 2, newitem)
                                            newitem = QTableWidgetItem(str(inc_id2))
                                            self.tableWidget.setItem(mynum + 1, 2, newitem)
                                            newitem = QTableWidgetItem(str(inc_s1))
                                            self.tableWidget.setItem(mynum, 3, newitem)
                                            newitem = QTableWidgetItem(str(inc_s2))
                                            self.tableWidget.setItem(mynum + 1, 3, newitem)
                                            newitem = QTableWidgetItem(str(inc_sj1))
                                            self.tableWidget.setItem(mynum, 4, newitem)
                                            newitem = QTableWidgetItem(str(inc_sj2))
                                            self.tableWidget.setItem(mynum + 1, 4, newitem)
                                            newitem = QTableWidgetItem(str(inc_a1))
                                            self.tableWidget.setItem(mynum, 5, newitem)
                                            newitem = QTableWidgetItem(str(inc_a2))
                                            self.tableWidget.setItem(mynum + 1, 5, newitem)
                                            newitem = QTableWidgetItem(str(inc_co1))
                                            self.tableWidget.setItem(mynum, 6, newitem)
                                            newitem = QTableWidgetItem(str(inc_co2))
                                            self.tableWidget.setItem(mynum + 1, 6, newitem)

                                            mynum = mynum + 2

                                            # 导出事故数据文件
                                            csvwriter.writerow([frame_index, video_time, img_name, (c4_x, c4_y),
                                                                '%.2f' % (s1 * pix2meter * 3.6),
                                                                '%.2f' % (s2 * pix2meter * 3.6), '%.2f' % speed_jump1,
                                                                '%.2f' % speed_jump2, '%.2f' % angle1, '%.2f' % angle2,
                                                                '%.2f' % Central_offset1,
                                                                '%.2f' % Central_offset2])
                    # 对已经消失的目标予以排除
                    for s in car_dic:
                        if car_dic["%d" % int(s)]["traced_frames"] > 0:
                            car_dic["%d" % int(s)]["traced_frames"] -= 1
                    for n in list(car_dic):
                        if car_dic["%d" % int(n)]["traced_frames"] == 0:
                            del car_dic["%d" % int(n)]

                    show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
                    self.label.setScaledContents(True)
                    self.label.setPixmap(QPixmap.fromImage(showImage))

                    show2 = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                    showImage2 = QImage(show2.data, show2.shape[1], show2.shape[0], QImage.Format_RGB888)
                    self.label_2.setScaledContents(True)
                    self.label_2.setPixmap(QPixmap.fromImage(showImage2))

                    show3 = QImage(image1.data,image1.shape[1],image1.shape[0],QImage.Format_RGB888)
                    self.label_3.setScaledContents(True)
                    self.label_3.setPixmap(QPixmap.fromImage(show3))

                    show4 = cv2.cvtColor(warped,cv2.COLOR_BGR2RGB)
                    showImage4 = QImage(show4.data, show4.shape[1], show4.shape[0], QImage.Format_RGB888)
                    self.label_4.setScaledContents(True)
                    self.label_4.setPixmap(QPixmap.fromImage(showImage4))

                    if writeVideo_flag:
                        out1.write(frame)
                        out2.write(bgr)
                        out3.write(image1)
                        out4.write(warped)
                        frame_index = frame_index + 1

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                    prev = next
                else:
                    break
            csvfile.close()
            video_capture.release()
            if writeVideo_flag:
                out1.release()
                out2.release()
                out3.release()
                out4.release()
            cv2.destroyAllWindows()
            self.player1 = QMediaPlayer()
            self.player2 = QMediaPlayer()
            self.player3 = QMediaPlayer()
            self.player4 = QMediaPlayer()
            self.player1.setVideoOutput(self.widget)
            self.player2.setVideoOutput(self.widget_2)
            self.player3.setVideoOutput(self.widget_3)
            self.player4.setVideoOutput(self.widget_4)
            self.player1.setMedia(QMediaContent(QUrl.fromLocalFile(strname2)))
            self.player1.play()
            self.player2.setMedia(QMediaContent(QUrl.fromLocalFile(strname5)))
            self.player2.play()
            self.player3.setMedia(QMediaContent(QUrl.fromLocalFile(strname3)))
            self.player3.play()
            self.player4.setMedia(QMediaContent(QUrl.fromLocalFile(strname4)))
            self.player4.play()

class Point(object):
    x = 0
    y = 0
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

if __name__ == '__main__':
    app = QApplication(sys.argv)
    myui = newui()
    global a, b
    a = 0
    b = 0
    coordinate = []
    myui.show()
    sys.exit(app.exec_())
