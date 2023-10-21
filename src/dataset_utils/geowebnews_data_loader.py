import os
import sys
import numpy as np
import json 
import math

import torch
from transformers import RobertaTokenizer, BertTokenizerFast
from torch.utils.data import Dataset
sys.path.append('/home/zekun/joint_model/src/datasets')
from dataset_loader import SpatialDataset

import pdb
np.random.seed(2333)


class GWN_ToponymDataset(SpatialDataset):
    def __init__(self, data_file_path,  tokenizer=None, max_token_len = 512, distance_norm_factor = 0.0001, spatial_dist_fill=10, mode = None ):
        
        if tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

        self.max_token_len = max_token_len
        self.spatial_dist_fill = spatial_dist_fill # should be normalized distance fill, larger than all normalized neighbor distance
        self.mode = mode
        
        self.read_file(data_file_path, mode)

        super(ToponymDataset, self).__init__(self.tokenizer , max_token_len , distance_norm_factor, sep_between_neighbors = True  )
        
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.mask_token_id  = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

        self.label_to_id = {'O': 0, 'B-topo':1, 'I-topo':2}
        self.id_to_label = {0:'O',1:'B-topo',2:'I-topo'}

    def read_file(self, data_file_path, mode):
        with open(data_file_path, 'r') as f:
            data = json.load(f)

        if mode == 'train':
            data = data[0:int(len(data) * 0.8)]
        elif mode == 'test':
            data = data[int(len(data) * 0.8):]
        elif mode is None: # use the full dataset (for mlm)
            pass
        else:
            raise NotImplementedError
        
        print('Dataset length ', len(data))
        self.data = data
        self.len_data = len(data)

    def get_offset_mappings(self, offset_mapping):
        flat_offset_mapping = np.array(offset_mapping).flatten()
        offset_mapping_dict_start = {}
        offset_mapping_dict_end = {}
        for idx in range(0,len(flat_offset_mapping),2):
            char_pos = flat_offset_mapping[idx]
            if char_pos == 0 and idx != 0:
                continue
            token_pos = idx//2 + 1 
            offset_mapping_dict_start[char_pos] = token_pos 
        for idx in range(1,len(flat_offset_mapping),2):
            char_pos = flat_offset_mapping[idx]
            if char_pos == 0 and idx != 0:
                break
            token_pos = (idx-1)//2 + 1 +1
            offset_mapping_dict_end[char_pos] = token_pos    

        return offset_mapping_dict_start, offset_mapping_dict_end

    def load_data(self, index):
        record = self.data[index]
        sentence = record['sentence']
        # print(len(sentence))

        # regular expression to break sentences 
        # (?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s

        toponyms = record['toponyms']

        input_tokens = self.tokenizer(sentence,  padding="max_length", max_length=self.max_token_len, truncation = False, return_offsets_mapping = True)
        input_ids = input_tokens['input_ids']

        offset_mapping_dict_start, offset_mapping_dict_end =  self.get_offset_mappings(input_tokens['offset_mapping'][1:-1]) 
        # labels = np.array([self.label_to_id['O'] for i in range(self.max_token_len)])
        labels = np.array([self.label_to_id['O'] for i in range(len(input_ids))]) #  initialize labels array with 0s
        labels[0] = -100 # set CLS and SEP to -100
        labels[-1] = -100

        for toponym in toponyms:
            start = toponym['start']
            end = toponym['end']
            if start not in offset_mapping_dict_start: # TODO: handle long sentences
                print('offset_mapping_dict_start', offset_mapping_dict_start)
                print('start', start, sentence, input_tokens['offset_mapping'][1:-1])
            if end not in offset_mapping_dict_end and end+1 not in offset_mapping_dict_end:
                print(len(sentence))
                print('end', end, sentence, input_tokens['offset_mapping'][1:-1])
            # token_start_idx, token_end_idx = offset_mapping_dict_start[start],offset_mapping_dict_end[end]
            try:
                token_start_idx, token_end_idx = offset_mapping_dict_start[start],offset_mapping_dict_end[end]
            except:
                token_start_idx, token_end_idx = offset_mapping_dict_start[start],offset_mapping_dict_end[end+1]
            assert token_start_idx < token_end_idx # can not be equal
        
            labels[token_start_idx + 1: token_end_idx ] = 2 
            labels[token_start_idx] = 1

        input_tokens['labels']  = labels 

        ret_dict = {}
        if len(input_ids) > self.max_token_len: 
            rand_start = np.random.randint(1, len(input_ids) - self.max_token_len ) # Do not include CLS and SEP [inclusive, exclusive)
            ret_dict['input_ids'] = torch.tensor([self.cls_token_id] + list(input_tokens['input_ids'][rand_start: rand_start + self.max_token_len -2]) + [self.sep_token_id])
            ret_dict['attention_mask'] = torch.tensor([1] + list(input_tokens['attention_mask'][rand_start: rand_start + self.max_token_len -2]) + [1])
            ret_dict['labels'] = torch.tensor([-100] + list(input_tokens['labels'][rand_start: rand_start + self.max_token_len -2]) + [-100])
        elif len(input_ids) < self.max_token_len:
            pad_len = self.max_token_len - len(input_ids)
            ret_dict['input_ids'] = torch.tensor(list(input_tokens['input_ids']) + [self.pad_token_id] * pad_len )
            ret_dict['attention_mask'] = torch.tensor(list(input_tokens['attention_mask']) + [0] * pad_len)
            ret_dict['labels'] = torch.tensor(list(input_tokens['labels']) + [-100] * pad_len)
        else:
            ret_dict['input_ids'] = torch.tensor(input_tokens['input_ids'])
            ret_dict['attention_mask'] = torch.tensor(input_tokens['attention_mask'])
            ret_dict['labels'] = torch.tensor(input_tokens['labels'] )

        ret_dict['sent_position_ids'] = torch.tensor(np.arange(0, self.max_token_len))
        ret_dict['norm_lng_list'] = torch.tensor([self.spatial_dist_fill for i in range(self.max_token_len)]).to(torch.float32)
        ret_dict['norm_lat_list'] = torch.tensor([self.spatial_dist_fill for i in range(self.max_token_len)]).to(torch.float32)
        ret_dict['token_type_ids'] =  torch.zeros(self.max_token_len).int() # 0 for nl data

        return ret_dict
        
    
    def __len__(self):
        return self.len_data

    def __getitem__(self, index):
        return self.load_data(index)