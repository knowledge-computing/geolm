#!/usr/bin/env python
# coding: utf-8

import sys
import os
import numpy as np
import pdb
import json
import scipy.spatial as sp
import argparse


import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm  # for our progress bar

sys.path.append('../../src/')
from datasets.dataset_loader import SpatialDataset
from models.spatial_bert_model import SpatialBertModel
from models.spatial_bert_model import SpatialBertConfig
from utils.find_closest import find_ref_closest_match, sort_ref_closest_match
from utils.common_utils import load_spatial_bert_pretrained_weights, get_spatialbert_embedding, get_bert_embedding, write_to_csv
from utils.baseline_utils import get_baseline_model
from transformers import BertModel
from transformers import AutoModel, AutoTokenizer
import pdb

from haversine import haversine, Unit


MODEL_OPTIONS = ['joint-base','joint-large', 'bert-base','bert-large','roberta-base','roberta-large',
'spanbert-base','spanbert-large',# 'luke-base','luke-large',
'simcse-bert-base','simcse-bert-large','simcse-roberta-base','simcse-roberta-large', 'simcse-base',
'sap-bert','mirror-bert']



def get_offset_mapping(nl_tokens):
    offset_mapping = nl_tokens['offset_mapping'][1:-1]
    flat_offset_mapping = np.array(offset_mapping).flatten()
    offset_mapping_dict_start = {}
    offset_mapping_dict_end = {}
    for idx in range(0,len(flat_offset_mapping),2):
        char_pos = flat_offset_mapping[idx]
        if char_pos == 0 and idx != 0:
            break
        token_pos = idx//2 + 1 
        offset_mapping_dict_start[char_pos] = token_pos 
    for idx in range(1,len(flat_offset_mapping),2):
        char_pos = flat_offset_mapping[idx]
        if char_pos == 0 and idx != 0:
            break
        token_pos = (idx-1)//2 + 1 +1
        offset_mapping_dict_end[char_pos] = token_pos

    return offset_mapping_dict_start, offset_mapping_dict_end

def get_nl_feature(text, gt_lat, gt_lon, start_span, end_span, model, model_name, tokenizer, spatial_dist_fill, device):

    # text = paragraph['text']
    # gt_lat = paragraph['lat']
    # gt_lon = paragraph['lon']

    # spans = paragraph['spans'] # TODO: can be improved
    # selected_span = spans[0]

    sentence_len = 512
    nl_tokens = tokenizer(text,  padding="max_length", max_length=sentence_len, truncation = True, return_offsets_mapping = True)
    offset_mapping_dict_start, offset_mapping_dict_end = get_offset_mapping(nl_tokens)
    if start_span not in offset_mapping_dict_start or end_span not in offset_mapping_dict_end:
        # pdb.set_trace()
        return None  # TODO: exceeds length. fix later

    token_start_idx = offset_mapping_dict_start[start_span]
    token_end_idx = offset_mapping_dict_end[end_span]

    nl_tokens['sent_position_ids'] = torch.tensor(np.array([np.arange(0, sentence_len)])).to(device)
    nl_tokens['norm_lng_list'] = torch.tensor([[0 for i in range(sentence_len)]]).to(torch.float32).to(device)
    nl_tokens['norm_lat_list'] = torch.tensor([[0 for i in range(sentence_len)]]).to(torch.float32).to(device)
    nl_tokens['token_type_ids'] = torch.zeros(1,sentence_len).int().to(device)
    entity_token_idx = torch.tensor([[token_start_idx, token_end_idx]]).to(device)
    
    if model_name == 'joint-base' or model_name=='joint-large' or model_name == 'simcse-base':
        nl_outputs = model(torch.tensor([nl_tokens['input_ids']]).to(device), 
                attention_mask = torch.tensor([nl_tokens['attention_mask']]).to(device), 
                sent_position_ids = nl_tokens['sent_position_ids'], 
                pivot_token_idx_list=entity_token_idx,
                position_list_x = nl_tokens['norm_lng_list'], 
                position_list_y = nl_tokens['norm_lat_list'] , 
                token_type_ids = nl_tokens['token_type_ids'])

        nl_entity_feature = nl_outputs.pooler_output
        nl_entity_feature = nl_entity_feature[0].detach().cpu().numpy()
    else:
        nl_outputs = model(torch.tensor([nl_tokens['input_ids']]).to(device),
                            attention_mask =torch.tensor([nl_tokens['attention_mask']]).to(device), 
                            token_type_ids = nl_tokens['token_type_ids'],
                            )
        embeddings = nl_outputs.last_hidden_state
        nl_entity_feature = embeddings[0][token_start_idx:token_end_idx] 
        nl_entity_feature = torch.mean(nl_entity_feature, axis = 0).detach().cpu().numpy() # (768, )
        # print(nl_entity_feature.shape)

    return nl_entity_feature

def get_geoname_features(geonames_cand, model, model_name, tokenizer, spatial_dist_fill, device, spatial_dataset):
    

    geo_feat_list = []
    geo_id_list = []
    geo_loc_list = []
    for gn_cand in geonames_cand:
        
        pivot_name = gn_cand['info']['name']
        pivot_pos = gn_cand['info']['geometry']['coordinates'] #(lng, lat)
        pivot_geonames_id = gn_cand['info']['geoname_id']


        neighbor_info = gn_cand['neighbor_info']
        neighbor_name_list = neighbor_info['name_list']
        neighbor_geometry_list = neighbor_info['geometry_list']
        

        geo_data = spatial_dataset.parse_spatial_context(pivot_name, pivot_pos, neighbor_name_list, neighbor_geometry_list, spatial_dist_fill )
        geo_data['token_type_ids'] = torch.ones( len(geo_data['pseudo_sentence'])).int() # type 1 for geo data 
        
        if model_name == 'joint-base' or model_name=='joint-large' or model_name == 'simcse-base':
            geo_outputs = model(geo_data['pseudo_sentence'].unsqueeze(0).to(device),
                            attention_mask = geo_data['attention_mask'].unsqueeze(0).to(device), 
                            sent_position_ids = geo_data['sent_position_ids'].unsqueeze(0).to(device),
                            pivot_token_idx_list= geo_data['pivot_token_idx'].unsqueeze(0).to(device),
                            position_list_x = geo_data['norm_lng_list'].unsqueeze(0).to(device),
                            position_list_y = geo_data['norm_lat_list'].unsqueeze(0).to(device),
                            token_type_ids = geo_data['token_type_ids'].unsqueeze(0).to(device),
                            )
            geo_feat = geo_outputs.pooler_output
            geo_feat = geo_feat[0].detach().cpu().numpy()

        else:
            # pdb.set_trace()
            if model_name == 'roberta-base':
                geo_outputs = model(geo_data['pseudo_sentence'].unsqueeze(0).to(device),
                            attention_mask = geo_data['attention_mask'].unsqueeze(0).to(device), 
                            )
            else:
                geo_outputs = model(geo_data['pseudo_sentence'].unsqueeze(0).to(device),
                                attention_mask = geo_data['attention_mask'].unsqueeze(0).to(device), 
                                token_type_ids = geo_data['token_type_ids'].unsqueeze(0).to(device),
                                )
            
            embeddings = geo_outputs.last_hidden_state
            geo_feat = embeddings[0][geo_data['pivot_token_idx'][0]:geo_data['pivot_token_idx'][1]] 
            geo_feat = torch.mean(geo_feat, axis = 0).detach().cpu().numpy() # (768, )
 

            # pivot_embed = pivot_embed[0].detach().cpu().numpy()

        # print(geo_feat.shape)
                
        geo_feat_list.append(geo_feat)
        geo_id_list.append(pivot_geonames_id)
        geo_loc_list.append({'lon':pivot_pos[0], 'lat':pivot_pos[1]})

    return geo_feat_list, geo_loc_list, geo_id_list


def wiktor_linking(out_path, query_data, geonames_dict, model, model_name, tokenizer, spatial_dist_fill, device, spatial_dataset):

    distance_list = []
    acc_count_at_161 = 0
    acc_count_total = 0
    correct_geoname_count = 0

    with open(out_path, 'w') as f:
        pass # flush


    # overall list, geodestic distance histogram  
    for query_name, paragraph_list in query_data.items():
        
        if query_name in geonames_dict:
            geonames_cand = geonames_dict[query_name]
            # print(query_name, len(paragraph_list), len(geonames_dict[query_name]))
            geoname_features, geonames_loc_list, geonames_id_list = get_geoname_features(geonames_cand, model, model_name, tokenizer, spatial_dist_fill, device, spatial_dataset)
        else:
            continue 
            # print(query_name, 'not in geonames_dict')
    
        samename_ret_list = []
        for paragraph in paragraph_list:
        #     cur_dict = {'text':text, 'feature':feature, 'url':url, 'country':country, 'lat':lat, 'lon':lon,
        # 'spans':spans}
            if 'url' in paragraph:
                wiki_url = paragraph['url']
            else: 
                wiki_url = None 

            text = paragraph['text']
            gt_lat = paragraph['lat']
            gt_lon = paragraph['lon']

            spans = paragraph['spans'] # TODO: can be improved
            if len(spans) == 0:
                # pdb.set_trace()
                continue 
            selected_span = spans[0]
            start_span, end_span = selected_span[0], selected_span[1]


            nl_feature = get_nl_feature(text, gt_lat, gt_lon, start_span, end_span, model, model_name, tokenizer, spatial_dist_fill, device)
            
            if nl_feature is None: continue 

            # nl_feature_shape:  torch.Size([1, 768])
            
            sim_matrix = 1 - sp.distance.cdist(np.array(geoname_features), np.array([nl_feature]), 'cosine')
        
            closest_match_geonames_id = sort_ref_closest_match(sim_matrix, geonames_id_list)
            closest_match_geonames_loc = sort_ref_closest_match(sim_matrix, geonames_loc_list)
                
            sorted_sim_matrix = np.sort(sim_matrix, axis = 0)[::-1] # descending order

            ret_dict = dict()
            ret_dict['pivot_name'] = query_name
            ret_dict['gt_loc'] = {'lon':paragraph['lon'], 'lat':paragraph['lat']}
            ret_dict['wiki_url'] = wiki_url
            ret_dict['sorted_match_geoname_id'] = [a[0] for a in closest_match_geonames_id]
            ret_dict['closest_match_geonames_loc'] = [a[0] for a in closest_match_geonames_loc]
            #ret_dict['sorted_match_des'] = [a[0] for a in closest_match_des]
            ret_dict['sorted_sim_matrix'] = [a[0] for a in sorted_sim_matrix]

            samename_ret_list.append(ret_dict)

            # print(ret_dict['gt_loc'], ret_dict['wiki_url'], ret_dict['closest_match_geonames_loc'])

            gt_loc = (float(paragraph['lat']), float(paragraph['lon']))
            pred_loc = ret_dict['closest_match_geonames_loc'][0]
            pred_loc = (pred_loc['lat'], pred_loc['lon'])
            error_dist = haversine(gt_loc, pred_loc)
            distance_list.append(error_dist)
            # pdb.set_trace()
            if error_dist < 161:
                acc_count_at_161 += 1

            acc_count_total+=1

            ret_dict['sorted_match_geoname_id'] = ret_dict['sorted_match_geoname_id']
            ret_dict['closest_match_geonames_loc'] = ret_dict['closest_match_geonames_loc']
            
            with open(out_path, 'a') as f:
                json.dump(ret_dict, f)
                f.write('\n')
    
    return {'distance_list':distance_list, 'acc_at_161:': 1.0*acc_count_at_161/acc_count_total}


def toponym_linking(out_path, query_data, geonames_dict, model, model_name, tokenizer, spatial_dist_fill, device, spatial_dataset):

    distance_list = []
    acc_count_at_161 = 0
    acc_count_total = 0
    correct_geoname_count = 0

    with open(out_path, 'w') as f:
        pass # flush

    for sample in query_data:
        #     cur_dict = {'sentence':sentence, 'toponyms':[]}
        text = sample['sentence']

        for toponym in sample['toponyms']:
            if 'geoname_id' not in toponym:
                continue # skip this sample in evaluation

            query_name = toponym['text'] 
            start_span = toponym['start']
            end_span = toponym['end']
            geoname_id = toponym['geoname_id']
            gt_lat = toponym['lat']
            gt_lon = toponym['lon']
            
            nl_feature = get_nl_feature(text, gt_lat, gt_lon, start_span, end_span, model, model_name, tokenizer, spatial_dist_fill, device)

            if nl_feature is None: continue

            if query_name in geonames_dict:
                geonames_cand = geonames_dict[query_name]
                # print(query_name, len(geonames_dict[query_name]))
                geoname_features, geonames_loc_list, geonames_id_list = get_geoname_features(geonames_cand, model, model_name, tokenizer, spatial_dist_fill, device, spatial_dataset)
                # pdb.set_trace()
                # print(geoname_features)
            else:
                continue 

            sim_matrix = 1 - sp.distance.cdist(np.array(geoname_features), np.array([nl_feature]), 'cosine')
        
            closest_match_geonames_id = sort_ref_closest_match(sim_matrix, geonames_id_list)
            closest_match_geonames_loc = sort_ref_closest_match(sim_matrix, geonames_loc_list)
                
            sorted_sim_matrix = np.sort(sim_matrix, axis = 0)[::-1] # descending order

            ret_dict = dict()
            ret_dict['pivot_name'] = query_name
            ret_dict['gt_loc'] = {'lon':gt_lon, 'lat':gt_lat}
            ret_dict['geoname_id'] = geoname_id
            ret_dict['sorted_match_geoname_id'] = [a[0] for a in closest_match_geonames_id]
            ret_dict['closest_match_geonames_loc'] = [a[0] for a in closest_match_geonames_loc]
            #ret_dict['sorted_match_des'] = [a[0] for a in closest_match_des]
            ret_dict['sorted_sim_matrix'] = [a[0] for a in sorted_sim_matrix]

            # samename_ret_list.append(ret_dict)

            # print(ret_dict['gt_loc'],  ret_dict['closest_match_geonames_loc'])

            gt_loc = (gt_lat, gt_lon)
            pred_loc = ret_dict['closest_match_geonames_loc'][0]
            pred_loc = (pred_loc['lat'], pred_loc['lon'])
            error_dist = haversine(gt_loc, pred_loc)
            distance_list.append(error_dist)

            if error_dist < 161:
                acc_count_at_161 += 1

            if str(ret_dict['sorted_match_geoname_id'][0]) == geoname_id:
                correct_geoname_count += 1

            acc_count_total+=1
            ret_dict['sorted_match_geoname_id'] = ret_dict['sorted_match_geoname_id']
            ret_dict['closest_match_geonames_loc'] = ret_dict['closest_match_geonames_loc']

            with open(out_path, 'a') as f:
                json.dump(ret_dict, f)
                f.write('\n')

    return {'distance_list':distance_list, 'acc@1': 1.0*correct_geoname_count/acc_count_total, 'acc_at_161:': 1.0*acc_count_at_161/acc_count_total}


def entity_linking_func(args):

    model_name = args.model_name
   
    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill= args.spatial_dist_fill
    sep_between_neighbors = True 
    spatial_bert_weight_dir = args.spatial_bert_weight_dir
    spatial_bert_weight_name = args.spatial_bert_weight_name
    if_no_spatial_distance = args.no_spatial_distance

    
    assert model_name in MODEL_OPTIONS


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    out_dir = args.out_dir

    print('out_dir', out_dir)
        
    if model_name == 'joint-base' or model_name == 'joint-large' or model_name =='simcse-base':
        if model_name == 'joint-base' or model_name == 'joint-large':
            tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        else:
            tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased')
        
        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance)
        
        config.vocab_size = tokenizer.vocab_size

        model = SpatialBertModel(config)

        model.to(device)
        model.eval()
        
        # load pretrained weights
        weight_path = os.path.join(spatial_bert_weight_dir, spatial_bert_weight_name)
        model = load_spatial_bert_pretrained_weights(model, weight_path)

    elif model_name in MODEL_OPTIONS: #'bert-base':
        model, tokenizer = get_baseline_model(model_name)
        # model.config.type_vocab_size=2
        model.to(device)
        model.eval()
    else:
        raise NotImplementedError

    spatial_dataset = SpatialDataset(tokenizer , max_token_len=512 ,  distance_norm_factor=distance_norm_factor, sep_between_neighbors = True)


    with open(args.query_dataset_path,'r') as f:
        query_data = json.load(f)

    geonames_dict = {}

    with open(args.ref_dataset_path,'r') as f:
        geonames_data = json.load(f)
        for info in geonames_data:
            key = next(iter(info))
            value = info[key]

            geonames_dict[key] = value

    if 'WikToR' in args.query_dataset_path:
        out_path  = os.path.join(out_dir, 'wiktor.json')
        eval_info = wiktor_linking(out_path, query_data, geonames_dict, model, model_name, tokenizer, spatial_dist_fill, device, spatial_dataset)
    elif 'lgl' in args.query_dataset_path or 'geowebnews' in args.query_dataset_path: 
        if 'lgl' in args.query_dataset_path: 
            out_path  = os.path.join(out_dir, 'lgl.json') 
        elif 'geowebnews' in args.query_dataset_path:
            out_path = os.path.join(out_dir, 'geowebnews.json') 

        eval_info = toponym_linking(out_path, query_data, geonames_dict, model, model_name, tokenizer, spatial_dist_fill, device, spatial_dataset)

    # print(distance_list)
    # print('acc_at_161:',1.0*acc_count_at_161/acc_count_total)

    with open(out_path, 'a') as f:
        json.dump(eval_info, f)
        f.write('\n')
        
    print(eval_info)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, default='joint-base')
    parser.add_argument('--query_dataset_path', type=str, default='../../data/WikToR.json')
    # parser.add_argument('--ref_dataset_path', type=str, default='../../data/geoname-ids-v3-part02.json')
    parser.add_argument('--ref_dataset_path', type=str, default='/home/zekun/datasets/geonames/geonames_for_wiktor/geoname-ids.json')

    parser.add_argument('--out_dir', type=str, default=None)

    parser.add_argument('--distance_norm_factor', type=float, default = 100)
    parser.add_argument('--spatial_dist_fill', type=float, default = 90000)
                       
    parser.add_argument('--no_spatial_distance',  default=False, action='store_true')

    parser.add_argument('--spatial_bert_weight_dir', type = str, default = None)
    parser.add_argument('--spatial_bert_weight_name', type = str, default = None)
                        
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    # out_dir not None, and out_dir does not exist, then create out_dir
    if args.out_dir is not None and not os.path.isdir(args.out_dir):
        os.makedirs(args.out_dir)

    entity_linking_func(args)

    # python3 link_geonames.py --out_dir='debug' --model_name='joint-base' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_run2/' --spatial_bert_weight_name='ep12_iter24000_0.0061.pth' 
    # ep14_iter88000_0.0039.pth
    # ep14_iter96000_0.0382.pth

    # python3 link_geonames.py --out_dir='debug' --model_name='bert-base' 

    # python3 link_geonames.py --model_name='bert-base' --query_dataset_path='/home/zekun/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --out_dir='baselines/bert-base'

    # CUDA_VISIBLE_DEVICES='1' python3 link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_run2/' --spatial_bert_weight_name='ep14_iter88000_0.0039.pth'

    # May

    # CUDA_VISIBLE_DEVICES='3' python3 link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/toponym_detection/geowebnews/GWN.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_geowebnews/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0505/' --spatial_bert_weight_name='ep0_iter80000_1.3164.pth' --out_dir='results'

    # CUDA_VISIBLE_DEVICES='3' python3 link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0505/' --spatial_bert_weight_name='ep0_iter80000_1.3164.pth' --out_dir='results'

    # CUDA_VISIBLE_DEVICES='0' python3 link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0505/' --spatial_bert_weight_name='ep1_iter52000_1.3994.pth' --out_dir='results'

    # CUDA_VISIBLE_DEVICES='0' python3 link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0506/' --spatial_bert_weight_name='ep0_iter48000_1.5495.pth' --out_dir='results-0506'

    # CUDA_VISIBLE_DEVICES='0' python3 link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/toponym_detection/lgl/lgl.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_lgl/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_run2/' --spatial_bert_weight_name='ep14_iter108000_0.0172.pth' --out_dir='results-run2'

    # CUDA_VISIBLE_DEVICES='0' python3 link_geonames.py --model_name='joint-base' --query_dataset_path='/home/zekun/toponym_detection/geowebnews/GWN.json' --ref_dataset_path='/home/zekun/datasets/geonames/geonames_for_geowebnews/geoname-ids.json' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_run2/' --spatial_bert_weight_name='ep14_iter108000_0.0172.pth' --out_dir='results-run2'

    # CUDA_VISIBLE_DEVICES='0' python3 link_geonames.py --model_name='joint-base' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0508/' --spatial_bert_weight_name='ep0_iter144000_0.5711.pth' --out_dir='results-run2'

    # ../../weights_base_0511/ep1_iter84000_0.5168.pth
    # CUDA_VISIBLE_DEVICES='0' python3 link_geonames.py --model_name='joint-base' --distance_norm_factor=100 --spatial_dist_fill=90000 --spatial_bert_weight_dir='/home/zekun/weights_base_0511/' --spatial_bert_weight_name='ep1_iter84000_0.5168.pth' --out_dir='debug'
    
 
    # CUDA_VISIBLE_DEVICES='1' python3 link_geonames.py --model_name='joint-base' --distance_norm_factor=100 --spatial_dist_fill=900 --spatial_bert_weight_dir='/data4/zekun/joint_model/weights_0517/' --spatial_bert_weight_name='ep5_iter04000_0.0486.pth' --out_dir='debug' --query_dataset_path='/data4/zekun/toponym_detection/geowebnews/GWN.json' --ref_dataset_path='/data4/zekun/geonames/geonames_for_geowebnews/geoname-ids.json'
    #  'acc@1': 0.23718439173680184, 'acc_at_161:': 0.31675592960979343}

if __name__ == '__main__':

    main()

    
