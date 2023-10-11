# Identification of Open-Set errors in Object Detectors through GradNorm

**This repository contains training and evaluation code for the thesis:
Exploring Gradient Based Methods for Identifying and Resolving Open-set
Errors**

*Brayth Tarlinton, Dimity Miller*

If you use this repository, please cite. 

**Contact**

Please contact [Brayth Tarlinton](braytarlinton@gmail.com) if you have any questions.

## Installation
*Developement environment (Recommended):*
\n   Python 3.7,
\n   Ubuntu 20.04,
\n   GPU

 
### Installing Requirements 
It is recommended that you use and install the following requirements into a python virtual environment.

*Install pytorch and mmcv:*

```bash
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.3.9 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
cd mmdetection
pip install -v -e .

```

## Datasets
The results in the thesis used the Pascal VOC and COCO datasets.

*Pascal VOC data can be downloaded from [here](http://host.robots.ox.ac.uk/pascal/VOC/).*
\nThe VOC2007 training/validation data, VOC2007 annotated test data, and VOC2012 training/validation data should be downloaded.

*COCO data can be downloaded from [here](https://cocodataset.org/#download).*
\nThe COCO 2017 train images, 2017 val images, and 2017 train/val annotations should be downloaded.

Move the datasets into `/datasets/data/` and verify it is in the following format:

 <br>
 
    └── datasets
        └── data
            ├── VOCdevkit
            |    ├── VOC2007               # containing train/val and test data from VOC2007
            |    |    ├── Annotations      # xml annotation for each image
            |    |    ├── ImageSets
            |    |    |   ├── Main         # train, val and test txt files
            |    |    |   └── ... 
            |    |    ├── JPEGImages       # 9,963 images
            |    |    └── ...                 
            |    └── VOC2012               # containing train and val data from VOC2012
            |         ├── Annotations      # xml annotation for each image
            |         ├── ImageSets
            |         |   ├── Main         # train and val txt files
            |         |   └── ... 
            |         ├── JPEGImages       # 17,125 images
            |         └── ...     
            └── coco
                ├── images
                |   ├── train2017          # 118,287 images
                |   └── val2017            # 5,019 images
                ├── annotations
                |   ├── instances_train2017.json 
                |   └── instances_val2017.json
                └── ... 

The dataset will need to be converted into their open-set varient, VOC-OS and COCO-OS for training and testing.

### Creating Open-Set Datasets
To create the open-set variants of each dataset, run the following commands:

```bash
cd datasets
python create_osdata.py --dataset voc
python create_osdata.py --dataset coco
```

This script will create 'closed-set' forms of VOC and COCO (i.e. VOC-CS and COCO-CS), and the original VOC and COCO will then be open-set datasets (i.e. VOC-OS and COCO-OS). For VOC, this is done by creating a new VOC2007CS and VOC2012CS folder with only closed-set images and closed-set annotations. For COCO, a new trainCS2017 and valCS2017 folder will be created, as well as new annotation files instances_trainCS2017.json and instances_valCS2017.json.                

## Pre-trained Models
You can download the Faster R-CNN pre-trained model weights at the following link: https://drive.google.com/file/d/1VOWfI8RGWhv_P-djDEUQAaBwX-8wwLGh/view?usp=sharing

The weights.zip file should be extracted into the mmdetection/ folder. Each pre-trained model folder contains a python script with the config used during training and a weights file 'latest.pth'.

<br>
 
    └── mmdetection
        └── weights
            ├── frcnn_CE_Voc0            # used for VOC standard baseline and ensemble baseline
            ├── frcnn_CE_Voc1            # used for VOC ensemble baseline
            ├── frcnn_CE_Voc2            # used for VOC ensemble baseline
            ├── frcnn_CE_Voc3            # used for VOC ensemble baseline
            ├── frcnn_CE_Voc4            # used for VOC ensemble baseline
            ├── frcnn_CACCE_Voc_A01      # used for GMM-Det, trained with CE and Anchor loss (with Anchor weight 0.1)
            ├── frcnn_CE_Coco0           # used for COCO standard baseline and ensemble baseline
            ├── frcnn_CE_Coco1           # used for COCO ensemble baseline
            ├── frcnn_CE_Coco2           # used for COCO ensemble baseline
            ├── frcnn_CE_Coco3           # used for COCO ensemble baseline
            ├── frcnn_CE_Coco4           # used for COCO ensemble baseline
            └── frcnn_CACCE_Coco_A005    # used for GMM-Det, trained with CE and Anchor loss (with Anchor weight 0.05)

## Evaluation
In order to evaluate the data, run the following command:

```bash
cd mmdetection
python3 test_data.py --dataset {dataset} --weights_directory {weights_directory} --save_name {save_name} --GradNorm {type_of_computation} --model {detection_model}
```
where:
* `dataset` : name of the dataset (*Options:* pascal-voc , coco )
* `weights_directory` : name of weights folder (location of .pt file)
* `save_name` : name of JSON file that contains results (output to 'results')
* `type_of_computation` : gradients used to compute GradNorm confidence value (*Options:* weights , weights_and_bias, bias)
* `detection_model` : name of object detector (*Options:* faster_rcnn , retina_net)

**Note:** Results may vary slightly due to updated packages 

## Training 
To train a model, the following configs can be used with the mmdet tools/train.py directory:

An example of training a retinanet model with the Pascal VOC Open-set dataset:

```bash
cd mmdetection

python3 tools/train.py configs/pascal_voc/retinanet_r50_fpn_1x_voc0712OS.py --gpus 1 --work-dir {working_directory}

```
* `working_directory` : where the weights will be save 

## Acknowledgement
mmdetection folder is from [open-mmlab/mmdetection](https://github.com/open-mmlab/mmdetection) github repository (with some minor additions and changes). Please acknowledge mmdetection if you use this respository.
The thesis is built upon the repository [openset_detection/](https://github.com/dimitymiller/openset_detection/) and results compare to the paper Uncertainty for Identifying Open-Set Errors in Visual Object Detection
by Dimity Miller, Niko Suenderhauf, Michael Milford, Feras Dayoub
