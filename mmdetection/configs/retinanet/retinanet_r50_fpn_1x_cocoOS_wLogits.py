_base_ = [
    '../_base_/models/retinanet_r50_fpn_cocoOS_wLogits.py',
    '../_base_/datasets/coco_detectionOS.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)