import os
import sys
import numpy as np
import time
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
import datetime
import _pickle as cPickle
import sed_eval
from utils import get_filename, inverse_scale
from pytorch_utils import forward
import config


class Evaluator(object):
    def __init__(self, model, data_generator, cuda=True):
        '''Evaluator to evaluate prediction performance. 
        
        Args: 
          model: object
          data_generator: object
          cuda: bool
        '''
        
        self.model = model
        self.data_generator = data_generator
        self.cuda = cuda
        
        self.frames_per_second = config.frames_per_second
        self.labels = config.labels
        self.in_domain_classes_num = len(config.labels) - 1
        self.all_classes_num = len(config.labels)
        self.idx_to_lb = config.idx_to_lb
        self.lb_to_idx = config.lb_to_idx

    def evaluate(self, data_type, iteration, max_iteration=None, verbose=False):
        '''Evaluate the performance. 
        
        Args: 
          data_type: 'train' | 'validate'
          max_iteration: None | int, maximum iteration to run to speed up evaluation
          verbose: bool
        '''

        generate_func = self.data_generator.generate_validate(
            data_type=data_type, 
            max_iteration=max_iteration)
        
        # Forward
        output_dict = forward(
            model=self.model, 
            generate_func=generate_func, 
            cuda=self.cuda, 
            return_target=True)
        file = 'wrong_list/'+ 'wrong_classification_' + str(iteration)
        output = output_dict['output']  # (audios_num, in_domain_classes_num)
        target = output_dict['target']  # (audios_num, in_domain_classes_num)
        filename = output_dict['filename']

        prob = np.exp(output)   # Subtask a, b use log softmax as output

        
        # Evaluate
        y_true = np.argmax(target, axis=-1)
        y_pred = np.argmax(prob, axis=-1)
#         print(y_pred)
        if data_type=='validate':
            for i in range(len(y_true)):
                if y_true[i] != y_pred[i]:
                    with open(file,'a') as f:
                        audioname = filename[i]
                        true_idx = str(y_true[i])
                        pred_idx = str(y_pred[i])
                        true_label = self.idx_to_lb[y_true[i]]
                        pred_label = self.idx_to_lb[y_pred[i]]
                        f.write(audioname+'\t'+true_idx+'\t'+true_label+'\t'+pred_idx+'\t'+pred_label+'\n')
                
    
        confusion_matrix = metrics.confusion_matrix(y_true, y_pred, labels=np.arange(self.in_domain_classes_num))
  
        classwise_accuracy = np.diag(confusion_matrix) \
            / np.sum(confusion_matrix, axis=-1)
        
        logging.info('Data type: {}'.format(data_type))
        

        logging.info('    Average ccuracy: {:.3f}'.format(np.mean(classwise_accuracy)))
        
        if verbose:
            classes_num = len(classwise_accuracy)
            for n in range(classes_num):
                logging.info('{:<20}{:.3f}'.format(self.labels[n], 
                    classwise_accuracy[n]))
                    
            logging.info(confusion_matrix)

        statistics = {
            'accuracy': classwise_accuracy, 
            'confusion_matrix': confusion_matrix}

        return statistics



class StatisticsContainer(object):
    def __init__(self, statistics_path):
        '''Container of statistics during training. 
        
        Args:
          statistics_path: string, path to write out
        '''
        self.statistics_path = statistics_path

        self.backup_statistics_path = '{}_{}.pickle'.format(
            os.path.splitext(self.statistics_path)[0], 
                datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

        self.statistics_dict = {'data': []}

    def append_and_dump(self, iteration, statistics):
        '''Append statistics to container and dump the container. 
        
        Args:
          iteration: int
          statistics: dict of statistics
        '''
        statistics['iteration'] = iteration
        self.statistics_dict['data'].append(statistics)

        cPickle.dump(self.statistics_dict, open(self.statistics_path, 'wb'))
        cPickle.dump(self.statistics_dict, open(self.backup_statistics_path, 'wb'))
        logging.info('    Dump statistics to {}'.format(self.statistics_path))