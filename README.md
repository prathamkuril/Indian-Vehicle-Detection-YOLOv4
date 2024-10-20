# YOLOv4 Indian Vehicle Detection

This repository contains a Jupyter notebook for training and running a YOLOv4 model to detect Indian vehicles using custom datasets. The notebook is designed to run on Google Colab, leveraging GPU resources for training and detection using the Darknet framework.

## Table of Contents
- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Usage](#usage)
  - [1. Mount Google Drive](#1-mount-google-drive)
  - [2. Install Darknet](#2-install-darknet)
  - [3. Prepare Dataset](#3-prepare-dataset)
  - [4. Train YOLOv4](#4-train-yolov4)
  - [5. Perform Detection](#5-perform-detection)
- [Results](#results)
- [Acknowledgments](#acknowledgments)

## Project Overview
The purpose of this notebook is to train a YOLOv4 object detection model to recognize different types of vehicles commonly found in India, such as cars, trucks, and motorbikes. YOLOv4 is a state-of-the-art object detection model known for its real-time performance. The notebook simplifies the training and detection process on a custom dataset using Google Colab.

## Requirements
To run this notebook successfully, you will need the following:
- A Google account for Colab access and Google Drive integration.
- A custom dataset in YOLO format, consisting of images and corresponding `.txt` annotation files.
- A GPU environment enabled in Colab to speed up training and inference.

## Usage

### 1. Mount Google Drive
You need to mount your Google Drive so that Colab can access your datasets and store the trained models.

```python
from google.colab import drive
drive.mount('/content/gdrive')

!ln -s /content/gdrive/My\ Drive/ /mydrive
```

This will create a symbolic link to your Google Drive folder, which can be accessed throughout the notebook.

### 2. Install Darknet
Next, clone the Darknet repository and compile it with GPU, OpenCV, and CUDNN support.

```bash
!git clone https://github.com/AlexeyAB/darknet
%cd darknet/

# Enable GPU, CUDNN, and OpenCV in Makefile
!sed -i 's/OPENCV=0/OPENCV=1/' Makefile
!sed -i 's/GPU=0/GPU=1/' Makefile
!sed -i 's/CUDNN=0/CUDNN=1/' Makefile
!sed -i 's/CUDNN_HALF=0/CUDNN_HALF=1/' Makefile

# Compile Darknet
!make
```

### 3. Prepare Dataset
Upload your dataset (images and annotations) to your Google Drive and unzip it within the Colab environment.

```bash
!unzip /mydrive/yolov4/obj.zip -d data/
```

Copy your custom configuration files, including `obj.names`, `obj.data`, and `yolov4-custom.cfg` from your Google Drive.

```bash
!cp /mydrive/yolov4/yolov4-custom.cfg cfg/
!cp /mydrive/yolov4/obj.names data/
!cp /mydrive/yolov4/obj.data data/
```

If necessary, preprocess the dataset using a Python script (optional step provided by `process.py`).

```bash
!cp /mydrive/yolov4/process.py .
!python process.py
```

### 4. Train YOLOv4

Download pre-trained convolutional weights for YOLOv4 and start training your custom model.

```bash
!wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.conv.137
```

Start training from scratch or resume from a checkpoint:

**Start from pre-trained weights:**
```bash
!./darknet detector train data/obj.data cfg/yolov4-custom.cfg yolov4.conv.137 -dont_show -map
```

**Resume from the last checkpoint:**
```bash
!./darknet detector train data/obj.data cfg/yolov4-custom.cfg /mydrive/yolov4/training/yolov4-custom_last.weights -dont_show -map
```

Monitor the GPU status to ensure proper utilization:

```bash
!nvidia-smi -L
```

### 5. Perform Detection
Once training is complete, you can test the model on new images by running detections and visualizing the results.

Modify the YOLOv4 configuration to change the batch size for detection:

```bash
!sed -i 's/batch=64/batch=1/' yolov4-custom.cfg
!sed -i 's/subdivisions=16/subdivisions=1/' yolov4-custom.cfg
```

Test the model on an image:

```bash
!./darknet detector test data/obj.data cfg/yolov4-custom.cfg /mydrive/yolov4/training/yolov4-custom_best.weights /mydrive/yolo_test_images/test_image.JPG -thresh 0.3
```

Use the provided `imShow()` function to display the detected output:

```python
def imShow(path):
  import cv2
  import matplotlib.pyplot as plt
  image = cv2.imread(path)
  height, width = image.shape[:2]
  resized_image = cv2.resize(image, (3*width, 3*height), interpolation=cv2.INTER_CUBIC)
  fig = plt.gcf()
  fig.set_size_inches(18, 10)
  plt.axis('off')
  plt.imshow(cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB))
  plt.show()

imShow('predictions.jpg')
```

## Results
After the training, the YOLOv4 model will be able to detect vehicles in Indian traffic scenes. Below is an example output from the model:

- **Sample Detection**:  
  ![Sample Detection](predictions.jpg)

You can evaluate the performance of the model using metrics such as mAP (mean Average Precision) by setting the `-map` flag during training.

## Acknowledgments
This project uses the Darknet framework by AlexeyAB. Special thanks to the YOLO community for continuous development and support. If you found this project helpful, please consider giving credit to the original Darknet repository and this project.

---

This `README.md` should now provide a clear, detailed guide for using the notebook you shared.
