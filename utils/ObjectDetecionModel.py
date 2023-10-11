import sys
import mmcv
import torch

from mmcv import Config
from mmcv.parallel import MMDataParallel 
from mmcv.runner import load_checkpoint,  wrap_fp16_model

sys.path.append('../mmdetection')
from mmdet.models import build_detector
from mmdet.datasets import replace_ImageToTensor


class ObjectDetecionModel:
    def __init__(self, model_type, dataset):
        self.faster_rcnn_config = {'pascal-voc': 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_voc0712OS_wLogits.py', 
                                   'coco': 'mmdetection/configs/faster_rcnn/faster_rcnn_r50_fpn_1x_cocoOS_wLogits.py'}
        
        self.retina_net_config = {'pascal-voc': 'mmdetection/configs/pascal_voc/retinanet_r50_fpn_1x_voc0712OS_wLogits.py', 
                                  'coco': 'mmdetection/configs/retinanet/retinanet_r50_fpn_1x_cocoOS_wLogits.py'}
        
        self.model_type = model_type

        if self.model_type == 'faster_rcnn':
            self.config = self.configure_model(self.faster_rcnn_config[dataset])

        elif self.model_type == 'retina_net':
            self.config = self.configure_model(self.retina_net_config[dataset])

        else:
            print('model type is invalid')

    def configure_model(self, config_file):
        """
            Returns a model configuration

            :param config:          Model configuration file 

            :return:                Model configuration
        """
                
        cfg = Config.fromfile(config_file)

        # import modules from string list.
        if cfg.get('custom_imports', None):
            from mmcv.utils import import_modules_from_strings
            import_modules_from_strings(**cfg['custom_imports'])
        # set cudnn_benchmark
        if cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True
        cfg.model.pretrained = None
        if cfg.model.get('neck'):
            if isinstance(cfg.model.neck, list):
                for neck_cfg in cfg.model.neck:
                    if neck_cfg.get('rfp_backbone'):
                        if neck_cfg.rfp_backbone.get('pretrained'):
                            neck_cfg.rfp_backbone.pretrained = None
            elif cfg.model.neck.get('rfp_backbone'):
                if cfg.model.neck.rfp_backbone.get('pretrained'):
                    cfg.model.neck.rfp_backbone.pretrained = None
        
        # in case the test dataset is concatenated
        if isinstance(cfg.data.testOS, dict):
            cfg.data.testOS.test_mode = True
        elif isinstance(cfg.data.testOS, list):
            for ds_cfg in cfg.data.testOS:
                ds_cfg.test_mode = True

        self.samples_per_gpu = cfg.data.testOS.pop('samples_per_gpu', 1)
        if self.samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.testOS.pipeline = replace_ImageToTensor(cfg.data.testOS.pipeline)

        return cfg
    
    def build_model(self, cfg, weights_directory, checkpoint, dataset):
        """
            Returns a model from the provided configuration, weights and checkpoint

            :param config:              Model configuration file 
            :param weights_directory:   Directory of weights file
            :param checkpoint:          the checkpoint for the given weights file

            :return:                    Model
        """
                
        print("#-- Building model --#")

        # build the model and load checkpoint
        model = build_detector(cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        method_list = [func for func in dir(model) if callable(getattr(model, func))]
        fp16_cfg = cfg.get('fp16', None)

        if fp16_cfg is not None:
            wrap_fp16_model(model)
        checkpoint = load_checkpoint(model, '{}/{}'.format(weights_directory, checkpoint), map_location='cpu')

        if 'CLASSES' in checkpoint['meta']:
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = dataset.CLASSES

        model = MMDataParallel(model, device_ids=[0])
        model.eval()

        return model