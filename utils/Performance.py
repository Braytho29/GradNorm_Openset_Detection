import json
import numpy as np
import scipy.stats as st
import sklearn.metrics
import sklearn.mixture as sm
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

class Performance:
    def __init__(self, associated_results_path, save_name):
        # Open associated detection results
        with open(f'{associated_results_path}/{save_name}.json', 'r') as f:
            associated_data = json.load(f)
        
        testType = np.asarray(associated_data['type'])
        testScores = np.asarray(associated_data['scores'])
        test_GradNorm_confs = np.asarray(associated_data["confs_gradnorm"])

        # Get results in terms of AUROC, and TPR at 5%, 10% and 20% FPR
        false_positive_rates = [0.05, 0.1, 0.2]

        # Get the performance results for base softmax score
        true_positive_known = testScores[testType == 0]
        false_positive_known = testScores[testType == 2]

        # Get the performance results for GradNorm
        GradNorm_true_positive_known = -test_GradNorm_confs[testType == 0]
        GradNorm_false_positive_known = -test_GradNorm_confs[testType == 2]

        self.softmax_score_results = self.summarise_performance(true_positive_known, 
                                                      false_positive_known, 
                                                      false_positive_rates, 
                                                      True, 
                                                      save_name + f' with uncertainty: Softmax score')

        self.GradNorm_results = self.summarise_performance(GradNorm_true_positive_known, 
                                                 GradNorm_false_positive_known, 
                                                 false_positive_rates, 
                                                 True, 
                                                 save_name + f' with uncertainty: GradNorm')


    def auroc_score(self, known_data, unknown_data):
        all_data = np.concatenate((known_data, unknown_data))
        labels = np.concatenate((np.zeros(len(known_data)), np.ones(len(unknown_data))))
        fpr, tpr, thresholds = sklearn.metrics.roc_curve(labels, all_data, pos_label = 0)  
        return fpr, tpr, sklearn.metrics.auc(fpr, tpr)

    def tpr_at_fpr(self, tpr, fpr, fpr_rate = 0.05):
        fpr_adjust = np.abs(np.array(fpr)-fpr_rate)
        fpr_idx = np.argmin(fpr_adjust)
        tpr_at_fpr = tpr[fpr_idx]

        return tpr_at_fpr, fpr[fpr_idx]
            
    def summarise_performance(self, known_data, unknown_data, fpr_rates = [], print_res = True, method_name = ''):
        results = {}

        fpr, tpr, auroc = self.auroc_score(known_data, unknown_data,)
        results['auroc'] = auroc
        results['fpr'] = list(fpr)
        results['tpr'] = list(tpr)

        spec_points = []
        for fpr_rate in fpr_rates:
            tprRate = self.tpr_at_fpr(tpr, fpr, fpr_rate)
            spec_points += [tprRate]

            results[f'tpr at fprRate {fpr_rate}'] = tprRate
        
        if print_res:
            print(f'Results for Method: {method_name}')
            print(f'------ AUROC: {round(auroc, 3)}')
            for point in spec_points:
                fp = point[1]
                tp = point[0]
                print(f'------ TPR at {round((100.*fp), 1)}FPR: {round((100.*tp), 1)}')

        return results