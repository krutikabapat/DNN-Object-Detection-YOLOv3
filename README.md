# DNN-Object-Detection-YOLO3
deep learning based object detection using YOLOv3 with OpenCV

## There are three main object detectors using deep learning:-

1. R-CNN (Selective Search), Fast R-CNN( Region proposed Network and R-CNN).  
2. Single shot detectors (SSD).  
3. YOLO.  

Both SSD and YOLO use one-stage detector strategy.  

## Requirements:-  

1. OpenCV 3.4.2 and above.  

## File structure:-

1. Models.sh (contains link to model files).  
2. yolo.py (python code for yolo).  
3. yolo.cpp (c++ code for yolo).  
5. lady.jpeg (input image file).  
6. video.mp4

## Usage:-  

For python:-  

1. <code> python3 yolo.py 'path to image' . </code>
2. <code> python3 yolo_video.py --video 'path to video' </code>  

For C++

1. <code> g++ yolo.cpp `pkg-config --cflags --libs opencv` </code>
## Run time:-

1. 0.7514 seconds  





