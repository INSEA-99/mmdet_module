Detection Module including mmdet, mmyolo
============

Usage
------------
1. [Setup](#Setup)
2. [Data Preparation](#Data-Preparation)
3. [Execution Instructions](#Execution-Instructions)
  
### Setup
```
conda create --name openmmlab python=3.8 -y
conda activate openmmlab
```
```
conda install pytorch torchvision -c pytorch
pip install -e .
```
  
### Data Preparation
Prepare COCO format dataset

```
coco_root_dir
    ├── annotations
    │   ├── train.json 
    │   └── val.json
    ├── train 
    │   ├── image1.jpg 
    │   ├── image2.jpg
    │        '''
    └── val
        ├── image3.jpg 
        ├── image4.jpg
             '''

```
you can use 'labelme_to_coco.py' to convert labelme format to coco format.
```
python labelme_to_coco.py root_dir classes --save_dir --split_ratio
python labelme_to_coco.py ./labelme_root_dir cat dog bird
```
root_dir must have labelme json files and images.  
```
root_dir
    ├── image1.jpg
    ├── image1.json
    ├── image2.jpg
    ├── image2.json
          '''
```
  
### Execution Instructions
create model
```
# model_type can be : mmdet, mmseg, mmpose
# mmdet-model_name can be :
# cascade_rcnn, dino-4scale, dino-5scale, rtmdet_tiny, rtmdet_x, yolox_l, yolov7_l, yolov8_l
model = create_model(model_type = "mmdet", 
                     model_name = "cascade_rcnn", 
                     data_root= '/home/inbada/mars_module/mars/mars/mmdetection/coco_format_data/', 
                     classes = ('hz', 'bz', 'chem'))
```
```
model.train()
model.test()
```
