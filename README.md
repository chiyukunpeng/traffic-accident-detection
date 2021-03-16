# traffic-accident-detection

##  Introduction

This project aims to infer traffic accident based on computer vision. Our method uses combined models based on `yolov3` and `deepsort`. Expriments show that our method outperforms state-of-the-art. More information please see [CSDN](https://blog.csdn.net/chiyukunpeng/article/details/103236840?spm=1001.2014.3001.5502)

## Quick Start

#### 1 Requirements
Python 3.8 or later with all requirements.txt dependencies installed, including torch>=1.6. To install run:
```
$ pip install -r requirements.txt
```
#### 2 Combine dataset
Please download dataset via mchar_data_list_0515.csv. To merge dataset run:
```
$ python image_merge.py --train_image_path   --val_image_path  --dst_image_path
```
#### 3 Create data.yaml
```
train: ../coco/images/train/ # 40k images
val: ../coco/images/val/  # 10k images

# number of classes
nc: 10

# class names
names: ['0','1','2','3','4','5','6','7','8','9']
```
#### 4 Create labels
To convert train.json to train.txt,please run:
```
$ python make_label.py --train_image_path  --val_image_path  --train_annotation_path --val_annotation_path --label_path
```
#### 5 Organize directory
```
---coco
    |---images
    |      |---train
    |      |---val
    |      |---test
    |---labels
           |---train
           |---val
---data
---models
---utils
---weights
```
#### 6 Train
Use the largest --batch-size your GPU allows (batch sizes shown for 16 GB devices). Pretrained weights are auto-downloaded from [Google Drive](https://drive.google.com/open?id=1Drs_Aiu7xx6S-ix95f9kNsA6ueKRpN2J).
```
$ python train.py --data ./data/coco.yaml --cfg ./models/yolov5s.yaml --weights ./weights/yolov5s.pt --batch_size 64  --epochs 20
```
Training losses and performance metrics are saved to Tensorboard and also to a **runs/exp0/results.txt** logfile. **results.txt** is plotted as **results.png** after training completes. Partially completed **results.txt** files can be plotted with **from utils.utils import plot_results; plot_results()**.
#### 7 Model merge(Optional)
Please run **result_merge.py**.

## References
https://github.com/ultralytics/yolov5

## Citation
If you find this project useful for your research, please cite:
```
@{street-view character recognition project,
author = {chen peng},
title = {SVCR},
website = {https://github.com/chiyukunpeng/street-view-character-recognition},
month = {August},
year = {2020}
}
```
