import os
import sys
import numpy as np
import json 
import math

import torch
from transformers import RobertaTokenizer, BertTokenizer
from torch.utils.data import Dataset
sys.path.append('/home/zekun/joint_model/src/dataset_utils')
from dataset_loader import SpatialDataset

import pdb
np.random.seed(2333)

class PbfMapDataset(SpatialDataset):
    def __init__(self, data_file_path,  tokenizer=None, max_token_len = 512, distance_norm_factor = 0.0001, spatial_dist_fill=10, 
        with_type = True, sep_between_neighbors = False, label_encoder = None, mode = None, num_neighbor_limit = None, random_remove_neighbor = 0.,type_key_str='class'):
        
        if tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

        self.max_token_len = max_token_len
        self.spatial_dist_fill = spatial_dist_fill # should be normalized distance fill, larger than all normalized neighbor distance
        self.with_type = with_type
        self.sep_between_neighbors = sep_between_neighbors
        self.label_encoder = label_encoder
        self.num_neighbor_limit = num_neighbor_limit
        self.read_file(data_file_path, mode)
        self.random_remove_neighbor = random_remove_neighbor
        self.type_key_str  = type_key_str # key name of the class type in the input data dictionary
 
        super(PbfMapDataset, self).__init__(tokenizer , max_token_len , distance_norm_factor, sep_between_neighbors )
        

    def read_file(self, data_file_path, mode):

        with open(data_file_path, 'r') as f:
            data = f.readlines()

        if mode == 'train':
            data = data[0:int(len(data) * 0.8)]
        elif mode == 'test':
            data = data[int(len(data) * 0.8):]
        elif mode is None: # use the full dataset (for mlm)
            pass
        else:
            raise NotImplementedError

        self.len_data = len(data) # updated data length
        self.data = data 

    def load_data(self, index):
        
        spatial_dist_fill = self.spatial_dist_fill
        line = self.data[index] # take one line from the input data according to the index

        line_data_dict = json.loads(line)

        # process pivot
        pivot_name = line_data_dict['info']['name']
        pivot_pos = line_data_dict['info']['geometry']['coordinates']

        
        neighbor_info = line_data_dict['neighbor_info']
        neighbor_name_list = neighbor_info['name_list']
        neighbor_geometry_list = neighbor_info['geometry_list']

        if self.random_remove_neighbor != 0:
            num_neighbors = len(neighbor_name_list)
            rand_neighbor = np.random.uniform(size = num_neighbors)

            neighbor_keep_arr = (rand_neighbor >= self.random_remove_neighbor) # select the neighbors to be removed
            neighbor_keep_arr = np.where(neighbor_keep_arr)[0]
            
            new_neighbor_name_list, new_neighbor_geometry_list = [],[]
            for i in range(0, num_neighbors):
                if i in neighbor_keep_arr:
                    new_neighbor_name_list.append(neighbor_name_list[i])
                    new_neighbor_geometry_list.append(neighbor_geometry_list[i])

            neighbor_name_list = new_neighbor_name_list
            neighbor_geometry_list = new_neighbor_geometry_list
        
        if self.num_neighbor_limit is not None:
            neighbor_name_list = neighbor_name_list[0:self.num_neighbor_limit]
            neighbor_geometry_list = neighbor_geometry_list[0:self.num_neighbor_limit]


        train_data = self.parse_spatial_context(pivot_name, pivot_pos, neighbor_name_list, neighbor_geometry_list, spatial_dist_fill )

        if self.with_type:
            pivot_type = line_data_dict['info'][self.type_key_str]
            train_data['pivot_type'] = torch.tensor(self.label_encoder.transform([pivot_type])[0]) # scalar, label_id

        # if 'ogc_fid' in line_data_dict['info']:
        #     train_data['ogc_fid'] = line_data_dict['info']['ogc_fid']

        return train_data

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.load_data(index)



class PbfMapDatasetMarginRanking(SpatialDataset):
    def __init__(self, data_file_path,  type_list = None, tokenizer=None, max_token_len = 512, distance_norm_factor = 0.0001, spatial_dist_fill=10, 
        sep_between_neighbors = False,  mode = None, num_neighbor_limit = None, random_remove_neighbor = 0., type_key_str='class'):
        
        if tokenizer is None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

        self.type_list  = type_list
        self.type_key_str  = type_key_str # key name of the class type in the input data dictionary
        self.max_token_len = max_token_len
        self.spatial_dist_fill = spatial_dist_fill # should be normalized distance fill, larger than all normalized neighbor distance
        self.sep_between_neighbors = sep_between_neighbors
        # self.label_encoder = label_encoder
        self.num_neighbor_limit = num_neighbor_limit
        self.read_file(data_file_path, mode)
        self.random_remove_neighbor = random_remove_neighbor
        self.mode = mode
        
 
        super(PbfMapDatasetMarginRanking, self).__init__(self.tokenizer , max_token_len , distance_norm_factor, sep_between_neighbors )
        

    def read_file(self, data_file_path, mode):

        with open(data_file_path, 'r') as f:
            data = f.readlines()

        if mode == 'train':
            data = data[0:int(len(data) * 0.8)]
        elif mode == 'test':
            data = data[int(len(data) * 0.8):]
            self.all_types_data = self.prepare_all_types_data()
        elif mode is None: # use the full dataset (for mlm)
            self.all_types_data = self.prepare_all_types_data()
            pass
        else:
            raise NotImplementedError

        self.len_data = len(data) # updated data length
        self.data = data 

    def prepare_all_types_data(self):
        type_list = self.type_list
        spatial_dist_fill = self.spatial_dist_fill
        type_data_dict = dict()
        for type_name in type_list:
            type_pos = [None, None] # use filler values
            type_data = self.parse_spatial_context(type_name, type_pos, pivot_dist_fill = 0.,
                neighbor_name_list = [], neighbor_geometry_list=[], spatial_dist_fill= spatial_dist_fill)
            type_data_dict[type_name] = type_data

        return type_data_dict

    def load_data(self, index):
        
        spatial_dist_fill = self.spatial_dist_fill
        line = self.data[index] # take one line from the input data according to the index

        line_data_dict = json.loads(line)

        # process pivot
        pivot_name = line_data_dict['info']['name']
        pivot_pos = line_data_dict['info']['geometry']['coordinates']

        
        neighbor_info = line_data_dict['neighbor_info']
        neighbor_name_list = neighbor_info['name_list']
        neighbor_geometry_list = neighbor_info['geometry_list']

        if self.random_remove_neighbor != 0:
            num_neighbors = len(neighbor_name_list)
            rand_neighbor = np.random.uniform(size = num_neighbors)

            neighbor_keep_arr = (rand_neighbor >= self.random_remove_neighbor) # select the neighbors to be removed
            neighbor_keep_arr = np.where(neighbor_keep_arr)[0]
            
            new_neighbor_name_list, new_neighbor_geometry_list = [],[]
            for i in range(0, num_neighbors):
                if i in neighbor_keep_arr:
                    new_neighbor_name_list.append(neighbor_name_list[i])
                    new_neighbor_geometry_list.append(neighbor_geometry_list[i])

            neighbor_name_list = new_neighbor_name_list
            neighbor_geometry_list = new_neighbor_geometry_list
        
        if self.num_neighbor_limit is not None:
            neighbor_name_list = neighbor_name_list[0:self.num_neighbor_limit]
            neighbor_geometry_list = neighbor_geometry_list[0:self.num_neighbor_limit]


        train_data = self.parse_spatial_context(pivot_name, pivot_pos, neighbor_name_list, neighbor_geometry_list, spatial_dist_fill )

        if 'ogc_fid' in line_data_dict['info']:
            train_data['ogc_fid'] = line_data_dict['info']['ogc_fid']

        # train_data['pivot_type'] = torch.tensor(self.label_encoder.transform([pivot_type])[0]) # scalar, label_id
        
        pivot_type = line_data_dict['info'][self.type_key_str]
        train_data['pivot_type'] = pivot_type

        if self.mode == 'train' :
            # postive class
            postive_name = pivot_type # class type string as input to tokenizer
            positive_pos = [None, None] # use filler values
            postive_type_data = self.parse_spatial_context(postive_name, positive_pos, pivot_dist_fill = 0.,
                neighbor_name_list = [], neighbor_geometry_list=[], spatial_dist_fill= spatial_dist_fill)
            train_data['positive_type_data'] = postive_type_data


            # negative class
            other_type_list = self.type_list.copy()
            other_type_list.remove(pivot_type)
            other_type = np.random.choice(other_type_list)
            negative_name = other_type 
            negative_pos = [None, None] # use filler values
            negative_type_data = self.parse_spatial_context(negative_name, negative_pos, pivot_dist_fill = 0.,
                neighbor_name_list = [], neighbor_geometry_list=[], spatial_dist_fill= spatial_dist_fill)
            train_data['negative_type_data'] = negative_type_data

        elif self.mode == 'test' or self.mode == None:
            # return data for all class types in type_list
            train_data['all_types_data'] = self.all_types_data

        else:
            raise NotImplementedError

        return train_data

    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.load_data(index)


