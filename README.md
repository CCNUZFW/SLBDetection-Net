# Official SLBDetection-net



## Testing


``` shell
python test.py --data data/slb.yaml --img 640 --batch 32 --conf 0.001 --iou 0.65 --device 0 --weights best.pt 
```

## Training

Data preparation

Please contact us for datasets.


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

## Inference

``` shell
python detect.py --weights [bset.pt] --conf 0.25 --img-size 640 --source [data.jpg]
```

Tested with: Python 3.7.13, Pytorch 1.12.0+cu113


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
