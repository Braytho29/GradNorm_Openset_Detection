import sys
import tqdm
import json
import numpy as np
from Dataset import *
from ObjectDetecionModel import *

sys.path.append('../mmdetection')
from mmdet.datasets import build_dataset
							
class AssociateData:
    def __init__(self, model, dataset, inference_results_path, save_name):

        print('#-- Starting Association Stage --#')
        # Thresholds for validating a detection
        self.score_threshold = 0.2
        self.iou_threshold = 0.5

        test_dataset = build_dataset(model.config.data.testOS)

        print('#-- Associating Test data --#')

        # Initalise JSON dictionary for saving associated detections
        self.associated_data = {'scores': [], 'type': [], 'logits': [], 'ious': [], 'confs_gradnorm': [], 'imname': []}

        # Open detection results 
        with open(f'{inference_results_path}/{save_name}.json', 'r') as f:
            test_detections = json.load(f)


        for image_idx, image in enumerate(tqdm.tqdm(test_dataset, total = len(test_dataset))):
            image_name = test_dataset.data_infos[image_idx]['filename']
            detections = np.asarray(test_detections[image_name])
            annotations = test_dataset.get_ann_info(image_idx)

            if len(detections) == 0:
                continue

            detection_logits = detections[:, :-7]
            detection_bboxes = detections[:, -7:-3]
            detection_scores = detections[:, -2]
            detection_confs_GradNorm = detections[:,-1]
            detections_predicted = np.argmax(detection_logits, axis = 1)      

            # Only consider detections that meet the score threshold
            score_mask = detection_scores >= self.score_threshold
            detections_dic = {'predictions': detections_predicted[score_mask], 
                              'scores': detection_scores[score_mask], 
                              'boxes': detection_bboxes[score_mask],
                              'logits': detection_logits[score_mask], 
                              'confs_gradnorm': detection_confs_GradNorm[score_mask],
                              'imname': image_name}

            self.associate_detection(detections_dic, annotations, dataset.num_classes)



    def associate_detection(self, detections, annotations, known_class):
        """
            Associate detections for a given image

            :param detections:              Dictionary of detections from a given image
            :param annotations:             List of annotations for each detection
            :param known_class:             Number of known classes in dataset 
        """

        # Get known annotations (final class is background)
        known_boxes = annotations['bboxes'][annotations['labels'] < known_class]
        known_labels = annotations['labels'][annotations['labels'] < known_class]

        # Get unknown annotations
        unknown_boxes =  annotations['bboxes'][annotations['labels'] > known_class]

        # Sort scores from most confident to least 
        sorted_scores = np.sort(detections['scores'])[::-1]
        sorted_idxes = np.argsort(detections['scores'])[::-1]

        detections_associated = [0]*len(detections['scores'])
        
        # Associate known class
        if len(known_boxes) > 0:
            known_ious = self.calculate_iou(detections['boxes'], known_boxes)
            detections_associated = self.associate_known_detection(detections, 
                                                                   known_ious, 
                                                                   known_labels, 
                                                                   sorted_scores, 
                                                                   sorted_idxes, 
                                                                   detections_associated)
            
        # If detection overlaps an ignored object, discard the detection
        if len(annotations['bboxes_ignore']) > 0 and np.sum(detections_associated) != len(detections_associated):
            ignored_ious = self.calculate_iou(detections['boxes'], annotations['bboxes_ignore'])
            detections_associated = self.associate_ignored_detection(ignored_ious, 
                                                                     sorted_scores, 
                                                                     sorted_idxes, 
                                                                     detections_associated)

        # Associate unknown class
        if len(unknown_boxes) > 0 and np.sum(detections_associated) != len(detections_associated):
            unknown_ious = self.calculate_iou(detections['boxes'], unknown_boxes)
            detections_associated = self.associate_unkown_detection(detections, 
                                                                    unknown_ious, 
                                                                    sorted_scores, 
                                                                    sorted_idxes,
                                                                    detections_associated)

    def calculate_iou(self, boxes1, boxes2):
        """
            Calculate the intersection over union between two bounding boxes

            :param bboxes1:         First bounding box
            :param bboxes2:         Second bounding box

            :return: iou            IoU value
        """
        def run(bboxes1, bboxes2):
            """
                Function taken from: 
                https://medium.com/@venuktan/vectorized-intersection-over-union-iou-in-numpy-and-tensor-flow-4fa16231b63d
            """

            x11, y11, x12, y12 = np.split(bboxes1, 4, axis=1)
            x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
            xA = np.maximum(x11, np.transpose(x21))
            yA = np.maximum(y11, np.transpose(y21))
            xB = np.minimum(x12, np.transpose(x22))
            yB = np.minimum(y12, np.transpose(y22))
            interArea = np.maximum((xB - xA + 1), 0) * np.maximum((yB - yA + 1), 0)
            boxAArea = (x12 - x11 + 1) * (y12 - y11 + 1)
            boxBArea = (x22 - x21 + 1) * (y22 - y21 + 1)
            iou = interArea / (boxAArea + np.transpose(boxBArea) - interArea)
            return iou
        return run(boxes1, boxes2) 

    def associate_known_detection(self, detections, known_ious, known_labels, 
                                  sorted_scores, sorted_idxes, detections_associated):
        """
            Associate known detections within a given image. 

            :param detections:              Dictionary of detections from a given image
            :param known_ious:              List of IoUs between detected bboxes and known object bboxes
            :param known_labels:            List of labels for known classes
            :param sorted_scores:           List of scores from most confident to least
            :param sorted_idxes:            List of indexes from the sorted scores
            :param detections_associated:   List that tracks which detections have been associated

            :return: detections_associated  Updated list of detections which have been associated
        """
                
        known_associated = [0]*len(known_labels)

        for score_idx, score in enumerate(sorted_scores):
            # If all known data has been associated, exit the loop
            if np.sum(known_associated) == len(known_associated):
                break

            detection_idx = sorted_idxes[score_idx]
            ious = known_ious[detection_idx]

            # Sort from greatest to lowest overlap
            sorted_iouIdxs = np.argsort(ious)[::-1]
            
            for iou_idx in sorted_iouIdxs:
                # Check that the known object has not already been detected
                if known_associated[iou_idx] == 1:
                    continue
                
                if ious[iou_idx] >= self.iou_threshold:
                    
                    # Mark detection as associated and as a known detection
                    known_associated[iou_idx] = 1
                    detections_associated[detection_idx] = 1
                    
                    # If predicted class is the same as the known label, mark as correctly classified
                    if detections['predictions'][detection_idx] == known_labels[iou_idx]:
                        self.append_dataholder(self.associated_data, 
                                               detections, 
                                               detection_idx, 
                                               score, 
                                               ious[iou_idx], 
                                               0)
                        
                    # If not the same, mark known class as missclassifed 
                    else:
                        self.append_dataholder(self.associated_data, 
                                               detections, 
                                               detection_idx, 
                                               score, 
                                               ious[iou_idx], 
                                               1)
                    
                else:
                    # If detection does not have an IoU greater than threshold, ignore it
                    break

        return detections_associated

    def associate_ignored_detection(self, ignored_ious, sorted_scores, sorted_idxes, detections_associated):
        """
            Ignore detections marked as ignored and update assocaited detections list as associated

            :param ignored_ious:            List of IoUs for ignored detections
            :param sorted_scores:           List of scores from most confident to least
            :param sorted_idxes:            List of indexes from the sorted scores
            :param detections_associated:   List that tracks which detections have been associated

            :return: detections_associated  Updated list of detections which have been associated
        """

        for score_idx, score in enumerate(sorted_scores):
            detection_idx = sorted_idxes[score_idx]

            # If detection has already been associated, skip it
            if detections_associated[detection_idx] == 1:
                continue

            # Sort IoUs from greatest to lowest overlap
            sorted_iou_idxs = np.argsort(ignored_ious[detection_idx])[::-1]

            for iou_idx in sorted_iou_idxs:
                if ignored_ious[detection_idx][iou_idx] >= self.iou_threshold:
                    # Mark detection as associated
                    detections_associated[detection_idx] = 1
                break
    
        return detections_associated
    
    def associate_unkown_detection(self, detections, unknown_ious, sorted_scores, sorted_idxes, detections_associated):
        """
            Associate unknown detections within a given image. 

            :param detections:              Dictionary of detections from a given image
            :param unknown_ious:            List of IoUs between detected bboxes and unknown object bboxes
            :param sorted_scores:           List of scores from most confident to least
            :param sorted_idxes:            List of indexes from the sorted scores
            :param detections_associated:   List that tracks which detections have been associated

            :return: detections_associated  Updated list of detections which have been associated
        """

        for score_idx, score in enumerate(sorted_scores):
            detection_idx = sorted_idxes[score_idx]

            # If detection has already been associated, skip it
            if detections_associated[detection_idx] == 1:
                continue

            ious = unknown_ious[detection_idx]

            # Sort from greatest to lowest overlap
            sorted_iouIdxs = np.argsort(ious)[::-1]

            for iou_idx in sorted_iouIdxs:
                if ious[iou_idx] >= self.iou_threshold:

                    # Mark detection as associated and as a unknown detection
                    detections_associated[detection_idx] = 1
                    self.append_dataholder(self.associated_data, detections, 
                                            detection_idx, score, ious[iou_idx], 2)
                    break
                else:
                    # If detection does not have an IoU greater than threshold, ignore it
                    break

        return detections_associated
        
    def append_dataholder(self, data_holder, detections, detection_idx, score, iou, detection_type):
        """
            Appended associated results to class dataholder

            :param data_holder:         Dictionary that contains associated results for each image in dataset
            :param detections:          List of detections within an image
            :param detection_idx:       Detection index within list
            :param score:               Detection score 
            :param iou:                 Detection iou
            :param detection_type       Type of detection (Known, Unknown)
        """
        
        data_holder['imname'] += [detections['imname']]
        data_holder['ious'] += [iou]
        data_holder['scores'] += [score]
        data_holder['logits'] += [list(detections['logits'][detection_idx])]
        data_holder['type'] += [detection_type]
        data_holder['confs_gradnorm'] += [detections['confs_gradnorm'][detection_idx]]
    