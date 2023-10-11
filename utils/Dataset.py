import sys
sys.path.append('../mmdetection')
from mmdet.datasets import build_dataset

class Dataset:
    def __init__(self, dataset: str, subsets):
        self.name = dataset
        self.number_of_known_classes = {'pascal-voc': 15, 'coco': 50} 
        self.num_classes = self.number_of_known_classes[dataset]

        self.voc_data = {'train12': "build_dataset(model.config.data.trainCS12)", 
                         'train07': "build_dataset(model.config.data.trainCS07)", 
                         'val': "build_dataset(model.config.data.val)",
                         'test': "build_dataset(model.config.data.testOS)"}
        
        self.coco_data = {'train': "build_dataset(model.config.data.trainCS)", 
                          'val': "build_dataset(model.config.data.val)", 
                          'test': "build_dataset(model.config.data.testOS)"}
        
        self.data_dic = {'pascal-voc': self.voc_data, 'coco': self.coco_data}
        self.data = self.data_dic[dataset]

        if subsets == 'all':
            self.subsets = ['train12', 'train07', 'val', 'test']
        else:
            self.subsets = [subsets]

            