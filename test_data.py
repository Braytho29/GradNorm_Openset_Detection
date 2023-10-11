import os
import sys
import json
import argparse
from pathlib import Path

sys.path.append("utils")
from Inference import Inference
from Dataset import Dataset
from AssociateData import AssociateData
from ObjectDetecionModel import ObjectDetecionModel
from Performance import Performance


def parse_args():
    parser = argparse.ArgumentParser(description='Test the data and save the raw detections')
    parser.add_argument('--dataset', default = 'pascal-voc', help='pascal-voc or coco')
    parser.add_argument('--subsets', default = 'test', help='train or val or test')
    parser.add_argument('--model', default = 'faster_rcnn', help='type of object detection model')
    parser.add_argument('--GradNorm', default = 'weights', help='type of uncertainty calculation for GradNorm')
    parser.add_argument('--weights_directory', default = None, help='directory of object detector weights')
    parser.add_argument('--checkpoint', default = 'latest.pth', help='what is the name of the object detector weights')
    parser.add_argument('--save_name', default = None, help='name to save results as')
    args = parser.parse_args()
    return args

def create_json(data, file_name, output_directory):
    """
        Creates a JSON file that contains all the results

        :param results:         Results of the detection attributes from the inference process
        :param dataset:         Dataset used for inference
        :param subset:          Subset of the dataset
        :param save_name:       Save name for the json file

        :return:                None
    """
            
    # Create JSON file and save to output directory
    JSON_data = json.dumps(data)

    # Check folders exist, if not, create it
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    f = open('{}/{}.json'.format(output_directory, file_name), 'w')
    f.write(JSON_data)
    f.close()

    print(f'#-- Inference results saved to: {output_directory}/{file_name}.json --#')

if __name__ == '__main__':
    args = parse_args()

    dataset = Dataset(args.dataset, args.subsets)
    object_detection_model = ObjectDetecionModel(args.model, args.dataset)


    BASE_WEIGHTS_FOLDER = os.path.dirname(os.path.realpath(__file__)) + f'/mmdetection/weights/{args.weights_directory}'
    BASE_RESULTS_FOLDER = os.path.dirname(os.path.realpath(__file__)) + '/results'

    # Inference the dataset using the specified object detection model
    inference = Inference(object_detection_model, dataset, BASE_WEIGHTS_FOLDER, args.checkpoint, args.GradNorm)

    # Save inference results to json file
    inference_results_dir = f'{BASE_RESULTS_FOLDER}/{args.model}/inference_results'
    create_json(inference.inference_results, args.save_name, inference_results_dir)

    # Associated known and unknown detections for testing
    associated_results = AssociateData(object_detection_model, dataset, inference_results_dir, args.save_name)

    # Save associated data
    associated_results_dir = f'{BASE_RESULTS_FOLDER}/{args.model}/associated_results'
    create_json(associated_results.associated_data, args.save_name, associated_results_dir)

    # Save performance results
    performance_results = Performance(associated_results_dir, args.save_name)
    performance_results_dir = f'{BASE_RESULTS_FOLDER}/{args.model}/performance_results'
    create_json(performance_results.softmax_score_results, args.save_name, performance_results_dir)
    create_json(performance_results.GradNorm_results, args.save_name, performance_results_dir)