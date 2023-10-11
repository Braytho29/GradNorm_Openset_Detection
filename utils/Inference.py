import sys
import tqdm
import numpy as np
from Dataset import *
from ObjectDetecionModel import *
sys.path.append('../mmdetection')
from mmdet.datasets import build_dataloader

class Inference:
    def __init__(self, model, dataset, weights_directory, checkpoint, GradNorm_type):             

        print('#-- Starting Inference Stage --#')
        for subset in dataset.subsets:
            # Build subset using mmcv inbuilt function
            subset_data = eval(dataset.data[subset])
            self.data_loader = self.create_dataloader(subset_data, model)
            self.built_model = model.build_model(model.config, weights_directory, checkpoint, subset_data)

            print(f"#-- Inferencing {subset} data --#")
            self.inference_results = self.inference_dataset(self.built_model, dataset.num_classes, 
                                                            self.data_loader, GradNorm_type, model.model_type)
        
    def create_dataloader(self, data, model):
        """
            Creates a data_loader for a given subset of a dataset

            :param data:            Data
            :param cfg:             Model configuration

            :return:                Data Loader
        """
        print('#-- Building data loader --#')
        data_loader = build_dataloader(data,
                                       samples_per_gpu=model.samples_per_gpu,
                                       workers_per_gpu=model.config.data.workers_per_gpu,
                                       dist=False,
                                       shuffle=False)
        return data_loader
    
    def inference_dataset(self, model, num_classes, data_loader, GradNorm_type, model_type):
        """
            Inferences the given subset and returns detection attributes and GradNorm score 
            for each image.

            :param subset:              Subset from a dataset
            :param model:               Object detection model
            :param data_loader:         Data loader for a given subset
            :param GradNorm_type:       Type of GradNorm calculation

            :return:                    Detection attributes for each image inferenced
        """
                
        number_of_images = len(data_loader.dataset)

        # Only detections with a max softmax above this score are considered valid
        score_threshold = 0.2 

        all_results = {}

        for image_id, data in enumerate(tqdm.tqdm(data_loader, total = number_of_images)):   
            image_name = data_loader.dataset.data_infos[image_id]['filename']
            all_results[image_name] = []
            all_detections = None     

            result = model(return_loss = False, rescale=True, **data)[0]  

            # Collect results from each object class and concatenate into a list of all the results
            for detection_id in range(num_classes):
                detections = result[detection_id]

                if len(detections) == 0:
                    continue
                
                # Attributes associated with image 
                bounding_boxes = detections[:, :5].detach().cpu().numpy()     # Bounding boxes associated with image
                distributions = detections[:, 5:]                             # Unnormalised distribution values for all classes associated with detection
                scores = detections[:, 4].detach().cpu().numpy()              # Confidence scores for all classes associated with detection
                GradNorm_confidences = []                                     # GradNorm list of confidences 
                scoresT = np.expand_dims(scores, axis=1)                      # Type assocaited with each score

                distributions_detached = distributions.detach().cpu().numpy()

                # Winning class must be class detection_id for this detection to be considered valid
                mask = np.argmax(distributions_detached, axis = 1) == detection_id     

                if np.sum(mask) == 0:
                    continue

                # Check thresholds are above the score cutoff (ensure detections above confidence score)
                image_detections = np.concatenate((distributions_detached, bounding_boxes, scoresT), 1)[mask]
                scores = scores[mask]
                masks_filtered = scores[::-1] >= score_threshold
                masks_filtered_indexs = np.where(masks_filtered)[0]

                # If no detections are above threshold, continue loop
                if np.sum(masks_filtered) == 0:
                    continue
                
                image_detections = image_detections[masks_filtered]
                model.zero_grad()

                # Apply GradNorm
                for detection in distributions[masks_filtered_indexs,:]:
                    
                    if model_type == "faster_rcnn":
                        GradNorm_confidence = self.faster_rcnn_GradNorm(model, detection, GradNorm_type)

                    if model_type == "retina_net":
                        GradNorm_confidence = self.retina_net_GradNorm(model, detection, GradNorm_type, detection_id)

                    GradNorm_confidences.append(GradNorm_confidence)        
                    model.zero_grad()
                image_detections = np.concatenate((image_detections, np.expand_dims(GradNorm_confidences, axis=1)), 1)

                # Initialise the variable 
                if all_detections is None:
                    all_detections = image_detections

                # Add to exisiting variable after first detection results
                else:
                    all_detections = np.concatenate((all_detections, image_detections))

            if all_detections is None:
                continue
            else:
                # Remove doubled-up detections -- this shouldn't really happen
                detections, idxes = np.unique(all_detections, return_index = True, axis = 0)

            all_results[image_name] = detections.tolist()

        return all_results

    def faster_rcnn_GradNorm(self, model, detection, GradNorm_type):
        """
            Computes the GradNorm confidence value for a given detection using Faster RCNN model parameters

            :param model:           object detection model
            :param detection:       detection result from object
            :param GradNorm_type:   GradNorm computation type

            :return:                GradNorm confidence value
        """

        # Softmax initialisation
        log_softmax = torch.nn.LogSoftmax(dim=-1).cuda()
        targets = torch.ones((1, len(detection))).cuda() 
        loss = torch.mean(torch.sum(-targets * log_softmax(detection), dim=-1))
        loss.backward(retain_graph=True)
        layer_weights = model.module.roi_head.bbox_head.fc_cls.weight.grad
        layer_bias = model.module.roi_head.bbox_head.fc_cls.bias.grad
        print(model.module.roi_head.bbox_head.fc_cls.weight)

        if GradNorm_type == 'weights':
            GradNorm_confidence = torch.mean(torch.abs(layer_weights)).cpu().numpy()

        elif GradNorm_type == 'weights_and_bias':
            GradNorm_confidence = torch.mean(torch.abs(layer_weights)).cpu().numpy() + torch.mean(torch.abs(layer_bias)).cpu().numpy()

        elif GradNorm_type == 'bias':
            GradNorm_confidence = torch.mean(torch.abs(layer_bias)).cpu().numpy()

        else:
            print("Invalid GradNorm computation type")
            GradNorm_confidence = None
    
        return GradNorm_confidence
    
    def retina_net_GradNorm(self, model, detection, GradNorm_type, detection_id):
        """
            Computes the GradNorm confidence value for a given detection using RetinaNet model parameters.
            Softmax is not required as it already done by the model

            :param model:           object detection model
            :param detection:       detection result from object
            :param GradNorm_type:   GradNorm computation type

            :return:                GradNorm confidence value
        """

        # Softmax initialisation
        targets = torch.ones((1, len(detection))).cuda() 
        targets[0][detection_id] = 0
        loss = torch.max(torch.sum(-targets * detection, dim=-1))
        loss.backward(retain_graph=True)

        layer_weights = model.module.bbox_head.retina_cls.weight.grad
        layer_bias = model.module.bbox_head.retina_cls.bias.grad

        if GradNorm_type == 'weights':
            GradNorm_confidence = torch.mean(torch.abs(layer_weights)).detach().cpu().numpy()

        elif GradNorm_type == 'weights_and_bias':
            GradNorm_confidence = torch.mean(torch.abs(layer_weights)).detach().cpu().numpy() + torch.mean(torch.abs(layer_bias)).detach().cpu().numpy()

        elif GradNorm_type == 'bias':
            GradNorm_confidence = torch.mean(torch.abs(layer_bias)).detach().cpu().numpy()

        else:
            print("Invalid GradNorm computation type")
            GradNorm_confidence = None

        return GradNorm_confidence