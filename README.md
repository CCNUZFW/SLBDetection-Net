# SLBDetection-Net: Towards closed-set and open-set student learning behavior detection in smart classroom of K-12 education

---
**SLBDetection-Net** is a cutting-edge method designed to improve the detection and representation of student learning behaviors in both closed-set and open-set classroom scenarios. Traditional methods often focus on behavior detection in limited or controlled environments, which can be inadequate for complex, real-world classroom settings.

![](image/fig1.jpg)

### Key Features:
   - **Multi-scale Focusing Key Information (MFKI)**: Focuses on accurately capturing learning behavior across different scales.
   - **Learning Behavior-aware Attention (LBA)**: An advanced mechanism for extracting key features and capturing complex characteristics of behaviors.
   - **LBA-Swin Transformer Block**: A specialized backbone network feature encoder that enhances the performance of SLBDetection-Net.
---
## Requirements
It's hard to reproduce the results without the same devices and environment. Although our code is highly compatible with mulitiple python environments, we strongly recommend that you create a new environment according to our settings.
   - matplotlib>=3.2.2
   - matplotlib>=3.2.2 
   - numpy>=1.18.5,<1.24.0 
   - opencv-python>=4.1.1 
   - Pillow>=7.1.2 
   - PyYAML>=5.3.1 
   - requests>=2.23.0 
   - scipy>=1.4.1 
   - torch>=1.7.0,!=1.12.0 
   - torchvision>=0.8.1,!=0.13.0 
   - tqdm>=4.41.0 
   - protobuf<4.21.3

## Data preparation
Please manage to acquire the original datasets and then run these scripts.

#### ClaBehavior
The original dataset can be acquired freely in the repository of [ClaBehavior](https://github.com/CCNUZFW/Student-behavior-detection-system). Please download ClaBehavior (coco format) and convert it to .xml format for experimentation.

#### Student Classroom Behavior dataset (SCB)
The raw data is available [here](https://github.com/Whiffe/SCB-dataset?tab=readme-ov-file), the database publisher provides several versions of the dataset, this paper uses version 1. please download and use it.


## Test
If you would like to evaluate these methods on a dataset, download the [weights](). Then run the following script.

``` shell
python test.py --data data/slb.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights best.pt 
```
Here, we provide inference results for example data.
![](image/fig2.jpg)

## Training
If you tend to evaluate these methods on your own dataset, please make sure to organize your data in the yolo format.

Single GPU training

``` shell
# train models
python train.py --workers 8 --device 0 --batch-size 32 --data [coco.yaml]l --img [640 640] --cfg [] --weights '' --name yolov7
```

Multiple GPU training

``` shell
# train models
python -m torch.distributed.launch --nproc_per_node 4 --master_port 9527 train.py --workers 8 --device 0,1,2,3 --sync-bn --batch-size 128 --data [coco.yaml] --img [640 640] --cfg [] --weights '' --name yolov7
```
## Representations
This project is intended for educational and research purposes only. When using this project, the user should ensure compliance with all applicable laws and regulations. The authors are not responsible for any legal issues or disputes arising from the use of this project.

## Acknowledgements

<details><summary> <b>Expand</b> </summary>

* [https://github.com/meituan/YOLOv6](https://github.com/meituan/YOLOv6)
* [https://github.com/WongKinYiu/PyTorch_YOLOv4](https://github.com/WongKinYiu/PyTorch_YOLOv4)
* [https://github.com/WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7)
* [https://github.com/ultralytics/yolov3](https://github.com/ultralytics/yolov3)
* [https://github.com/ultralytics/yolov5](https://github.com/ultralytics/yolov5)
* [https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn](https://pytorch.org/vision/stable/_modules/torchvision/models/detection/faster_rcnn.html#fasterrcnn_resnet50_fpn)
* [https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet](https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet)
</details>
