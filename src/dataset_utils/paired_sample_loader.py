import os
import sys
import numpy as np
import json 
import math

import torch
from transformers import RobertaTokenizer, BertTokenizer, BertTokenizerFast
from torch.utils.data import Dataset
sys.path.append('/home/zekun/joint_model/src/datasets')
from dataset_loader import SpatialDataset

import pdb
np.random.seed(2333)

class JointDataset(SpatialDataset):
    def __init__(self, geo_file_path, nl_file_path, placename_to_osmid_path, tokenizer=None, max_token_len = 512, distance_norm_factor = 0.0001, spatial_dist_fill=10, 
        sep_between_neighbors = False, label_encoder = None, if_rand_seq=False, type_key_str='class'):
        
        if tokenizer is None:
            self.tokenizer = BertTokenizerFast.from_pretrained('bert-base-cased')
        else:
            self.tokenizer = tokenizer

        self.max_token_len = max_token_len
        self.spatial_dist_fill = spatial_dist_fill # should be normalized distance fill, larger than all normalized neighbor distance
        self.sep_between_neighbors = sep_between_neighbors
        self.label_encoder = label_encoder
        self.read_placename2osm_dict(placename_to_osmid_path) # to prepare hard negative samples
        self.read_geo_file(geo_file_path)
        self.read_nl_file(nl_file_path)
        self.type_key_str = type_key_str # key name of the class type in the input data dictionary
        self.if_rand_seq = if_rand_seq
 
        super(JointDataset, self).__init__(self.tokenizer , max_token_len , distance_norm_factor, sep_between_neighbors )
        
        self.sep_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.sep_token)
        self.cls_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.cls_token)
        self.mask_token_id  = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        self.pad_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.pad_token)

    def read_placename2osm_dict(self, placename_to_osmid_path):
        with open(placename_to_osmid_path, 'r') as f:
            placename2osm_dict = json.load(f)
        self.placename2osm_dict = placename2osm_dict

    def read_geo_file(self, geo_file_path):

        with open(geo_file_path, 'r') as f:
            data = f.readlines()

        self.len_geo_data = len(data) # updated data length
        self.geo_data = data 
    
    def read_nl_file(self, nl_file_path):
        with open(nl_file_path, 'r') as f:
            nl_data = json.load(f)
        
        self.nl_data = nl_data

    def prepare_nl_data(self, pivot_osm_id):

        nl_sample_dict = self.nl_data[pivot_osm_id]
        sentences = nl_sample_dict['sentence']
        subject_index_list = nl_sample_dict['subject_index_list']

        sample_idx = np.random.randint(len(sentences))
        sent = sentences[sample_idx]
        subject_schar, subject_tchar = subject_index_list[sample_idx] # start and end index in character

        nl_tokens = self.tokenizer(sent,  padding="max_length", max_length=self.max_token_len, truncation = True, return_offsets_mapping = True)
        pseudo_sentence = nl_tokens['input_ids']

        rand = np.random.uniform(size = self.max_token_len)  
        
        mlm_mask_arr = (rand <0.15) & (np.array(pseudo_sentence) != self.cls_token_id) & (np.array(pseudo_sentence) != self.sep_token_id) & (np.array(pseudo_sentence) != self.pad_token_id)

        token_mask_indices = np.where(mlm_mask_arr)[0]
        
        masked_token_input = [self.mask_token_id if i in token_mask_indices else pseudo_sentence[i] for i in range(0, self.max_token_len)]

        
        offset_mapping = nl_tokens['offset_mapping'][1:-1]
        flat_offset_mapping = np.array(offset_mapping).flatten()
        offset_mapping_dict_start = {}
        offset_mapping_dict_end = {}
        for idx in range(0,len(flat_offset_mapping),2):
            char_pos = flat_offset_mapping[idx]
            if char_pos == 0 and idx != 0:
                break
            token_pos = idx//2  + 1 
            offset_mapping_dict_start[char_pos] = token_pos 
        for idx in range(1,len(flat_offset_mapping),2):
            char_pos = flat_offset_mapping[idx]
            if char_pos == 0 and idx != 0:
                break
            token_pos = (idx-1)//2 + 1 +1
            offset_mapping_dict_end[char_pos] = token_pos    

        if subject_schar not in offset_mapping_dict_start or subject_tchar not in offset_mapping_dict_end:
            print(pivot_osm_id, sample_idx)
            return self.prepare_nl_data(pivot_osm_id) # a work-around, TODO: fix this


        if offset_mapping_dict_start[subject_schar] == offset_mapping_dict_end[subject_tchar]:
            print('\n')
            print(offset_mapping_dict_start, offset_mapping_dict_end)
            print(subject_schar, subject_tchar)
            print(sent)
            print(pseudo_sentence)
            print(self.tokenizer.convert_ids_to_tokens(pseudo_sentence))
            print('\n')
        # token end index is exclusive
        token_start_idx, token_end_idx = offset_mapping_dict_start[subject_schar],offset_mapping_dict_end[subject_tchar]
        assert token_start_idx < token_end_idx # can not be equal

        
        train_data = {}
        train_data['masked_input'] = torch.tensor(masked_token_input)
        train_data['pivot_token_idx'] = torch.tensor([token_start_idx, token_end_idx])

        train_data['pseudo_sentence'] = torch.tensor(pseudo_sentence)
        train_data['sent_len'] = torch.tensor(np.sum(np.array(nl_tokens['attention_mask']) == 1)) # pseudo sentence length including CLS and SEP token
        train_data['attention_mask'] = torch.tensor(nl_tokens['attention_mask'])
        train_data['sent_position_ids'] = torch.tensor(np.arange(0, len(pseudo_sentence)))
        # train_data['norm_lng_list'] = torch.tensor([self.spatial_dist_fill for i in range(len(pseudo_sentence))]).to(torch.float32)
        # train_data['norm_lat_list'] = torch.tensor([self.spatial_dist_fill for i in range(len(pseudo_sentence))]).to(torch.float32)
        train_data['norm_lng_list'] = torch.tensor([0 for i in range(len(pseudo_sentence))]).to(torch.float32)
        train_data['norm_lat_list'] = torch.tensor([0 for i in range(len(pseudo_sentence))]).to(torch.float32)
        train_data['token_type_ids'] =  torch.zeros(len(pseudo_sentence)).int() # 0 for nl data

        return train_data

    def load_data(self, geo_line_data_dict):
        
        # process pivot
        pivot_name = geo_line_data_dict['info']['name']
        pivot_pos = geo_line_data_dict['info']['geometry']['coordinates']
        pivot_osm_id = geo_line_data_dict['info']['osm_id']

        neighbor_info = geo_line_data_dict['neighbor_info']
        neighbor_name_list = neighbor_info['name_list']
        neighbor_geometry_list = neighbor_info['geometry_list']
        # print(neighbor_geometry_list)

        train_data = {}
        train_data['geo_data'] = self.parse_spatial_context(pivot_name, pivot_pos, neighbor_name_list, neighbor_geometry_list, self.spatial_dist_fill )
        train_data['geo_data']['token_type_ids'] = torch.ones( len(train_data['geo_data']['pseudo_sentence'])).int() # type 1 for geo data 
        train_data['geo_data']['sent_len'] =  torch.sum((train_data['geo_data']['attention_mask']) == 1) # pseudo sentence length including CLS and SEP token
        train_data['nl_data'] = self.prepare_nl_data(pivot_osm_id)

        train_data['concat_data'] = {}
        nl_data_len = train_data['nl_data']['sent_len']
        geo_data_len = train_data['geo_data']['sent_len']

        if nl_data_len + geo_data_len <= self.max_token_len: 
            # if the total length is smaller than max_token_len, take the full sentence (remove [CLS] before geo sentence)
            # print (train_data['nl_data']['masked_input'][:nl_data_len].shape, train_data['geo_data']['masked_input'][1 : self.max_token_len - nl_data_len + 1].shape)
            train_data['concat_data']['masked_input'] = torch.cat((train_data['nl_data']['masked_input'][:nl_data_len] ,train_data['geo_data']['masked_input'][1 : self.max_token_len - nl_data_len + 1]))
            train_data['concat_data']['attention_mask'] = torch.cat((train_data['nl_data']['attention_mask'][:nl_data_len] ,train_data['geo_data']['attention_mask'][1 : self.max_token_len - nl_data_len + 1]))
            train_data['concat_data']['sent_position_ids'] =  torch.cat((train_data['nl_data']['sent_position_ids'][:nl_data_len] ,train_data['geo_data']['sent_position_ids'][1 : self.max_token_len - nl_data_len + 1]))
            train_data['concat_data']['pseudo_sentence'] = torch.cat((train_data['nl_data']['pseudo_sentence'][:nl_data_len] ,train_data['geo_data']['pseudo_sentence'][1 : self.max_token_len - nl_data_len + 1]))
            train_data['concat_data']['token_type_ids'] = torch.cat((train_data['nl_data']['token_type_ids'][:nl_data_len] ,train_data['geo_data']['token_type_ids'][1 : self.max_token_len - nl_data_len + 1]))
            
        else:
            # otherwise, 
            if nl_data_len <= self.max_token_len / 2 : 
                # if the nl_data_len is <=  0.5 * max_token_len, then truncate geodata 
                # concat geo data , remove [CLS] from geo_data
                # SEP alrady added at the end of nl sentence after tokenization
                # print(train_data['geo_data']['masked_input'][1 : self.max_token_len - nl_data_len ].shape, torch.tensor([self.sep_token_id]).shape)
                train_data['concat_data']['masked_input'] = torch.cat((train_data['nl_data']['masked_input'][:nl_data_len] , train_data['geo_data']['masked_input'][1 : self.max_token_len - nl_data_len ] , torch.tensor([self.sep_token_id])))
                train_data['concat_data']['attention_mask'] = torch.cat((train_data['nl_data']['attention_mask'][:nl_data_len] , train_data['geo_data']['attention_mask'][1 : self.max_token_len - nl_data_len ] , torch.tensor([0])))
                train_data['concat_data']['sent_position_ids'] = torch.cat((train_data['nl_data']['sent_position_ids'][:nl_data_len] , train_data['geo_data']['sent_position_ids'][1 : self.max_token_len - nl_data_len ] , torch.tensor([self.max_token_len - nl_data_len])))
                train_data['concat_data']['pseudo_sentence'] = torch.cat((train_data['nl_data']['pseudo_sentence'][:nl_data_len] , train_data['geo_data']['pseudo_sentence'][1 : self.max_token_len - nl_data_len ] , torch.tensor([self.sep_token_id])))
                train_data['concat_data']['token_type_ids'] = torch.cat((train_data['nl_data']['token_type_ids'][:nl_data_len] , train_data['geo_data']['token_type_ids'][1 : self.max_token_len - nl_data_len ] , torch.tensor([1])))
                
            else:
                
                train_data['concat_data']['masked_input'] = torch.cat((train_data['nl_data']['masked_input'][:self.max_token_len // 2 - 1], torch.tensor([self.sep_token_id]) , train_data['geo_data']['masked_input'][1 : self.max_token_len//2 ], torch.tensor([self.sep_token_id])))
                train_data['concat_data']['attention_mask'] = torch.cat((train_data['nl_data']['attention_mask'][:self.max_token_len // 2 - 1], torch.tensor([1]) , train_data['geo_data']['attention_mask'][1 : self.max_token_len//2 ], torch.tensor([0])))
                train_data['concat_data']['sent_position_ids'] = torch.cat((train_data['nl_data']['sent_position_ids'][:self.max_token_len // 2 - 1], torch.tensor([self.max_token_len // 2 - 1]) , train_data['geo_data']['sent_position_ids'][1 : self.max_token_len//2 ], torch.tensor([self.max_token_len//2])))
                train_data['concat_data']['pseudo_sentence'] = torch.cat((train_data['nl_data']['pseudo_sentence'][:self.max_token_len // 2 - 1], torch.tensor([self.sep_token_id]) , train_data['geo_data']['pseudo_sentence'][1 : self.max_token_len//2 ], torch.tensor([self.sep_token_id])))
                train_data['concat_data']['token_type_ids'] = torch.cat((train_data['nl_data']['token_type_ids'][:self.max_token_len // 2 - 1], torch.tensor([0]) , train_data['geo_data']['token_type_ids'][1 : self.max_token_len//2 ], torch.tensor([1])))

                # print('c', train_data['concat_data']['masked_input'].shape, train_data['concat_data']['attention_mask'].shape, train_data['concat_data']['sent_position_ids'].shape, 
                # train_data['concat_data']['sent_position_ids'].shape, train_data['concat_data']['pseudo_sentence'].shape, train_data['concat_data']['token_type_ids'].shape )


        train_data['concat_data']['norm_lng_list'] =  torch.tensor([self.spatial_dist_fill for i in range(self.max_token_len)]).to(torch.float32)
        train_data['concat_data']['norm_lat_list'] =  torch.tensor([self.spatial_dist_fill for i in range(self.max_token_len)]).to(torch.float32)

        return train_data

    def __len__(self):
        return self.len_geo_data

    def __getitem__(self, index):
        spatial_dist_fill = self.spatial_dist_fill

        if self.if_rand_seq:
            # randomly take samples, ignoring the index
            line = self.geo_data[np.random.randint(self.len_geo_data)]
        else:
            line = self.geo_data[index] # take one line from the input data according to the index

        geo_line_data_dict = json.loads(line)

        while geo_line_data_dict['info']['osm_id'] not in self.nl_data:
            line = self.geo_data[np.random.randint(self.len_geo_data)]
            geo_line_data_dict = json.loads(line)

        return self.load_data(geo_line_data_dict)
