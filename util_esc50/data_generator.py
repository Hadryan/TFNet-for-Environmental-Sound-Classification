import numpy as np
import h5py
import csv
import time
import logging
import os
import glob
import matplotlib.pyplot as plt
import logging

from utils import read_metadata, sparse_to_categorical
import config


class Base(object):
    
    def __init__(self):
        '''Base class for data generator
        '''
        pass

    def load_hdf5(self, hdf5_path):
        '''Load hdf5 file. 
        
        Returns:
          data_dict: dict of data, e.g.:
            {'audio_name': np.array(['a.wav', 'b.wav', ...]), 
             'feature': (audios_num, frames_num, mel_bins)
             'target': (audios_num,), 
             ...}
        '''
        data_dict = {}
        
        with h5py.File(hdf5_path, 'r') as hf:
            data_dict['filename'] = np.array(
                [filename.decode() for filename in hf['filename'][:]])

            data_dict['feature'] = hf['feature'][:].astype(np.float32)
            
            if 'category' in hf.keys():
                data_dict['target'] = np.array(
                    [self.lb_to_idx[category.decode()] \
                        for category in hf['category'][:]])

            if 'category' in hf.keys():
                data_dict['category'] = np.array(
                    [category.decode() \
                        for category in hf['category'][:]])
            if 'fold' in hf.keys():
                data_dict['fold'] = np.array(
                    [fold \
                        for fold in hf['fold'][:]])
            
        return data_dict



class DataGenerator(Base):
    
    def __init__(self, feature_hdf5_path, train_csv, validate_csv, holdout_fold, batch_size, seed=1234):
        '''Data generator for training and validation. 
        
        Args:
          feature_hdf5_path: string, path of hdf5 feature file
          train_csv: string, path of train csv file
          validate_csv: string, path of validate csv file
          holdout_fold: set 1 for development and none for training 
              on all data without validation
          scalar: object, containing mean and std value
          batch_size: int
          seed: int, random seed
        '''

        self.batch_size = batch_size
        self.random_state = np.random.RandomState(seed)
        
        # self.classes_num = classes_num
        self.in_domain_classes_num = len(config.labels)
        self.all_classes_num = len(config.labels)
        self.lb_to_idx = config.lb_to_idx
        self.idx_to_lb = config.idx_to_lb
        
        # Load training data
        load_time = time.time()
        
        self.data_dict = self.load_hdf5(feature_hdf5_path)
        
        train_meta = read_metadata(train_csv)
        validate_meta = read_metadata(validate_csv)

        self.train_audio_indexes = self.get_audio_indexes(
            train_meta, self.data_dict, holdout_fold, 'train')
            
        self.validate_audio_indexes = self.get_audio_indexes(
            validate_meta, self.data_dict, holdout_fold, 'validate')
            
        if holdout_fold == 'none':
            self.train_audio_indexes = np.concatenate(
                (self.train_audio_indexes, self.validate_audio_indexes), axis=0)
                
            self.validate_audio_indexes = np.array([])
        
        logging.info('Load data time: {:.3f} s'.format(time.time() - load_time))
        logging.info('Training audio num: {}'.format(len(self.train_audio_indexes)))            
        logging.info('Validation audio num: {}'.format(len(self.validate_audio_indexes)))
        
        self.random_state.shuffle(self.train_audio_indexes)
        self.pointer = 0
        
    def get_audio_indexes(self, meta_data, data_dict, holdout_fold, data_type):
        '''Get train or validate indexes. 
        '''
        audio_indexes = []
        
        for name in meta_data['filename']:
            loct = np.argwhere(data_dict['filename'] == name)
            
            if len(loct) > 0:
                index = loct[0, 0]
                label = self.idx_to_lb[self.data_dict['target'][index]]
                if holdout_fold != 'none':
                    if data_type == 'train':
                        audio_indexes.append(index)
                    elif data_type == 'validate':
                        audio_indexes.append(index)
                else:
                    audio_indexes.append(index)
            
        return np.array(audio_indexes)
        
    def generate_train(self):
        '''Generate mini-batch data for training. 
        
        Returns:
          batch_data_dict: dict containing audio_name, feature and target
        '''

        while True:
            # Reset pointer
            if self.pointer >= len(self.train_audio_indexes):
                self.pointer = 0
                self.random_state.shuffle(self.train_audio_indexes)

            # Get batch audio_indexes
            batch_audio_indexes = self.train_audio_indexes[
                self.pointer: self.pointer + self.batch_size]
                
            self.pointer += self.batch_size

            batch_data_dict = {}

            batch_data_dict['filename'] = \
                self.data_dict['filename'][batch_audio_indexes]
            
            batch_feature = self.data_dict['feature'][batch_audio_indexes]
            batch_data_dict['feature'] = batch_feature
            
            sparse_target = self.data_dict['target'][batch_audio_indexes]
            batch_data_dict['target'] = sparse_to_categorical(
                sparse_target, self.in_domain_classes_num)
            
            yield batch_data_dict
            
            
    def generate_validate(self, data_type, max_iteration=None):
        '''Generate mini-batch data for training. 
        
        Args:
          data_type: 'train' | 'validate'
          max_iteration: int, maximum iteration to validate to speed up validation
        
        Returns:
          batch_data_dict: dict containing audio_name, feature and target
        '''
        
        batch_size = self.batch_size
        
        if data_type == 'train':
            audio_indexes = np.array(self.train_audio_indexes)
        elif data_type == 'validate':
            audio_indexes = np.array(self.validate_audio_indexes)
        else:
            raise Exception('Incorrect argument!')
            
            
        iteration = 0
        pointer = 0
        
        while True:
            if iteration == max_iteration:
                break

            # Reset pointer
            if pointer >= len(audio_indexes):
                break

            # Get batch audio_indexes
            batch_audio_indexes = audio_indexes[pointer: pointer + batch_size]                
            pointer += batch_size
            iteration += 1

            batch_data_dict = {}

            batch_data_dict['filename'] = \
                self.data_dict['filename'][batch_audio_indexes]
            
            batch_feature = self.data_dict['feature'][batch_audio_indexes]
            batch_data_dict['feature'] = batch_feature
            
            sparse_target = self.data_dict['target'][batch_audio_indexes]
            batch_data_dict['target'] = sparse_to_categorical(
                sparse_target, self.all_classes_num)

            yield batch_data_dict

            