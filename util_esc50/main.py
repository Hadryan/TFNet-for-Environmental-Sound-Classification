import os
import sys
import numpy as np
import argparse
import h5py
import math
import time
import logging
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from SpecAugment import spec_augment_pytorch
from utils import (create_folder, get_filename, create_logging, mixup_data, mixup_criterion)
from data_generator import DataGenerator
from net import TFNet
from losses import nll_loss
from evaluate import Evaluator, StatisticsContainer
from pytorch_utils import move_data_to_gpu, forward
import config


def train(args, i):
    '''Training. Model will be saved after several iterations. 
    
    Args: 
      dataset_dir: string, directory of dataset
      workspace: string, directory of workspace
      holdout_fold: '1' | 'none', set 1 for development and none for training 
          on all data without validation
      model_type: string, e.g. 'Cnn_9layers_AvgPooling'
      batch_size: int
      cuda: bool
      mini_data: bool, set True for debugging on a small part of data
    '''
    
    # Arugments & parameters
    dataset_dir = args.dataset_dir
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    batch_size = args.batch_size
    cuda = args.cuda and torch.cuda.is_available()
    mini_data = args.mini_data
    filename = args.filename
    audio_num = config.audio_num
    mel_bins = config.mel_bins
    frames_per_second = config.frames_per_second
    max_iteration = None      # Number of mini-batches to evaluate on training data
    reduce_lr = True
    in_domain_classes_num = len(config.labels)
    
    # Paths
    if mini_data:
        prefix = 'minidata_'
    else:
        prefix = ''
    
    train_csv = os.path.join(sys.path[0], 'fold'+str(i)+'_train.csv')
        
    validate_csv = os.path.join(sys.path[0], 'fold'+str(i)+'_test.csv')
                
    feature_hdf5_path = os.path.join(workspace, 'features', 
        '{}logmel_{}frames_{}melbins.h5'.format(prefix, frames_per_second, mel_bins))
        
    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        '{}logmel_{}frames_{}melbins.h5'.format(prefix, frames_per_second, mel_bins), 
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_folder(checkpoints_dir)

    validate_statistics_path = os.path.join(workspace, 'statistics', filename, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins),
        'holdout_fold={}'.format(holdout_fold), 
        model_type, 'validate_statistics.pickle')
    
    create_folder(os.path.dirname(validate_statistics_path))
    
    logs_dir = os.path.join(workspace, 'logs', filename, args.mode, 
        '{}logmel_{}frames_{}melbins'.format(prefix, frames_per_second, mel_bins), 
        'holdout_fold={}'.format(holdout_fold), model_type)
    create_logging(logs_dir, 'w')
    logging.info(args)

    if cuda:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # Model
    Model = eval(model_type)
   
    model = Model(in_domain_classes_num, activation='logsoftmax')
    loss_func = nll_loss

    if cuda:
        model.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)
    # Data generator
    data_generator = DataGenerator(
        feature_hdf5_path=feature_hdf5_path, 
        train_csv=train_csv, 
        validate_csv=validate_csv, 
        holdout_fold=holdout_fold, 
        batch_size=batch_size)
    
    # Evaluator
    evaluator = Evaluator(
        model=model, 
        data_generator=data_generator, 
        cuda=cuda)
    
    # Statistics
    validate_statistics_container = StatisticsContainer(validate_statistics_path)

    train_bgn_time = time.time()
    iteration = 0
    
    # Train on mini batches
    for batch_data_dict in data_generator.generate_train():
        
        # Evaluate
        if iteration % 100 == 0 and iteration >= 1000:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            
            train_statistics = evaluator.evaluate(data_type='train', iteration= iteration,
                                                  max_iteration=None, verbose=False)
            
            if holdout_fold != 'none':
                validate_statistics = evaluator.evaluate(data_type='validate', 
                                                         iteration= iteration, max_iteration=None, verbose=False)
                validate_statistics_container.append_and_dump(iteration, validate_statistics)

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()

#         Save model
        if iteration % 100 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.state_dict(), 
                'optimizer': optimizer.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
            
        # Reduce learning rate
        if reduce_lr and iteration % 100 == 0 and iteration > 0:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.9
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            if key in ['feature', 'target']:
                batch_data_dict[key] = move_data_to_gpu(batch_data_dict[key], cuda)
                
        if iteration % 3 == 0:
            # Train 
            for i in range(audio_num):
                model.train() 
                data, target_a, target_b, lam = mixup_data(x=batch_data_dict['feature'][:, i, :, :, :], y=batch_data_dict['target'], alpha=0.2)
                batch_output = model(data)
        #         batch_output = model(batch_data_dict['feature'])
                # loss
#                 loss = loss_func(batch_output, batch_data_dict['target'])
                loss = mixup_criterion(loss_func, batch_output, target_a, target_b, lam)

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if iteration % 3 == 1:
            # Train 
            for i in range(audio_num):
                model.train() 
                batch_output = model(batch_data_dict['feature'][:, i, :, :, :])
                # loss
                loss = loss_func(batch_output, batch_data_dict['target'])

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        if iteration % 3 == 2:
            # Train 
            for i in range(audio_num):
                model.train() 
                data = spec_augment_pytorch.spec_augment(batch_data_dict['feature'][:, i, :, :, :], using_frequency_masking=True,
                                                 using_time_masking=True)
                batch_output = model(data)
                # loss
                loss = loss_func(batch_output, batch_data_dict['target'])

                # Backward
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
        # Stop learning
        if iteration == 4500:
            break
            
        iteration += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', 'none'], required=True, help='Set 1 for development and none for training on all data without validation.')
    parser_train.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--audio_num', type=int, default=4)
    parser_train.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    # Inference validation data
    parser_inference_validation = subparsers.add_parser('inference_validation')
    parser_inference_validation.add_argument('--dataset_dir', type=str, required=True, help='Directory of dataset.')
    parser_inference_validation.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_inference_validation.add_argument('--holdout_fold', type=str, choices=['1'], required=True)
    parser_inference_validation.add_argument('--model_type', type=str, required=True, help='E.g., Cnn_9layers_AvgPooling.')
    parser_inference_validation.add_argument('--iteration', type=int, required=True, help='Load model of this iteration.')
    parser_inference_validation.add_argument('--batch_size', type=int, required=True)
    parser_inference_validation.add_argument('--cuda', action='store_true', default=False)
    parser_inference_validation.add_argument('--audio_num', type=int, default=4)
    parser_inference_validation.add_argument('--visualize', action='store_true', default=False, help='Visualize log mel spectrogram of different sound classes.')
    parser_inference_validation.add_argument('--mini_data', action='store_true', default=False, help='Set True for debugging on a small part of data.')

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        for i in range(5):
            train(args, i+1)

    else:
        raise Exception('Error argument!')