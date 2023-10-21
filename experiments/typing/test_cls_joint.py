import os
import sys
from transformers import RobertaTokenizer, BertTokenizer, BertTokenizerFast
from tqdm import tqdm  # for our progress bar
from transformers import AdamW

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F

sys.path.append('/home/zekun/joint_model/src')
from models.spatial_bert_model import SpatialBertModel
from models.spatial_bert_model import SpatialBertConfig
from models.spatial_bert_model import  SpatialBertForMaskedLM, SpatialBertForSemanticTyping
from datasets.osm_sample_loader import PbfMapDataset
from datasets.const import *
from transformers.models.bert.modeling_bert import BertForMaskedLM

from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics import precision_recall_fscore_support
import numpy as np
import argparse 
from sklearn.preprocessing import LabelEncoder
import pdb


DEBUG = False


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


def testing(args):

    max_token_len = args.max_token_len
    batch_size = args.batch_size
    num_workers = args.num_workers
    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill=args.spatial_dist_fill
    with_type = args.with_type
    sep_between_neighbors = args.sep_between_neighbors
    checkpoint_path = args.checkpoint_path
    if_no_spatial_distance = args.no_spatial_distance

    bert_option = args.bert_option


 
    if args.num_classes == 9:
        # london_file_path = '/home/zekun/spatial_bert/spatial_bert/experiments/semantic_typing/data/sql_output/osm-point-london-typing.json'
        # california_file_path = '/home/zekun/spatial_bert/spatial_bert/experiments/semantic_typing/data/sql_output/osm-point-california-typing.json'
        london_file_path = '/home/zekun/datasets/semantic_typing/data/sql_output/osm-point-london-typing.json'
        california_file_path = '/home/zekun/datasets/semantic_typing/data/sql_output/osm-point-california-typing.json'
        TYPE_LIST = CLASS_9_LIST
        type_key_str  = 'class'
    elif args.num_classes == 74:
        london_file_path = '../../semantic_typing/data/sql_output/osm-point-london-typing-ranking.json'
        california_file_path = '../../semantic_typing/data/sql_output/osm-point-california-typing-ranking.json'
        TYPE_LIST = CLASS_74_LIST
        type_key_str = 'fine_class'
    else:
        raise NotImplementedError

    if bert_option == 'bert-base':
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance, num_semantic_types=len(TYPE_LIST))
    elif bert_option == 'bert-large':
        # tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance, hidden_size = 1024, intermediate_size = 4096, num_attention_heads=16, num_hidden_layers=24,num_semantic_types=len(TYPE_LIST))
    else:
        raise NotImplementedError


    config.vocab_size = 28996 
    model = SpatialBertForSemanticTyping(config)


    #model.load_state_dict(bert_model.state_dict() , strict = False) # load sentence position embedding weights as well
        

    label_encoder = LabelEncoder()
    label_encoder.fit(TYPE_LIST)
    #model.load_state_dict(torch.load('../weights/mlm_mem_ep0.pth'))
    

    london_dataset = PbfMapDataset(data_file_path = london_file_path, 
                                        tokenizer = tokenizer, 
                                        max_token_len = max_token_len, 
                                        distance_norm_factor = distance_norm_factor, 
                                        spatial_dist_fill = spatial_dist_fill, 
                                        with_type = with_type,
                                        type_key_str = type_key_str,
                                        sep_between_neighbors = sep_between_neighbors, 
                                        label_encoder = label_encoder,
                                        mode = 'test')

    california_dataset = PbfMapDataset(data_file_path = california_file_path, 
                                            tokenizer = tokenizer, 
                                            max_token_len = max_token_len, 
                                            distance_norm_factor = distance_norm_factor, 
                                            spatial_dist_fill = spatial_dist_fill, 
                                            with_type = with_type,
                                            type_key_str = type_key_str,
                                            sep_between_neighbors = sep_between_neighbors,
                                            label_encoder = label_encoder,
                                            mode = 'test')

    test_dataset = torch.utils.data.ConcatDataset([london_dataset, california_dataset])


    
    test_loader = DataLoader(test_dataset, batch_size= batch_size, num_workers=num_workers,
                            shuffle=False, pin_memory=True, drop_last=False)
    

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device(device)), strict = True) #, strict = False) # # load sentence position embedding weights as well

    model.eval()



    print('start testing...')

    
    # setup loop with TQDM and dataloader
    loop = tqdm(test_loader, leave=True)


    mrr_total = 0.
    prec_total = 0.
    sample_cnt = 0

    gt_list = []
    pred_list = []

    for batch in loop:

        input_ids = batch['pseudo_sentence'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        position_list_x = batch['norm_lng_list'].to(device)
        position_list_y = batch['norm_lat_list'].to(device)
        sent_position_ids = batch['sent_position_ids'].to(device)

        labels = batch['pivot_type'].to(device)
        entity_token_idx = batch['pivot_token_idx'].to(device)
        token_type_ids = torch.ones(input_ids.shape[0],512).int().to(device)

        outputs = model(input_ids, attention_mask = attention_mask, sent_position_ids = sent_position_ids,
            position_list_x = position_list_x, position_list_y = position_list_y, 
            labels = labels, token_type_ids = token_type_ids, pivot_token_idx_list=entity_token_idx)


        onehot_labels = F.one_hot(labels, num_classes=len(TYPE_LIST))
        
        gt_list.extend(onehot_labels.cpu().detach().numpy())
        pred_list.extend(outputs.logits.cpu().detach().numpy())

        mrr = label_ranking_average_precision_score(onehot_labels.cpu().detach().numpy(), outputs.logits.cpu().detach().numpy())
        mrr_total += mrr * input_ids.shape[0] 
        sample_cnt += input_ids.shape[0]

    precisions, recalls, fscores, supports = precision_recall_fscore_support(np.argmax(np.array(gt_list),axis=1), np.argmax(np.array(pred_list),axis=1), average=None)

    precision, recall, f1, _ = precision_recall_fscore_support(np.argmax(np.array(gt_list),axis=1), np.argmax(np.array(pred_list),axis=1), average='micro')
    print('precisions:\n', ["{:.3f}".format(prec) for prec in precisions])
    print('recalls:\n', ["{:.3f}".format(rec) for rec in recalls])
    print('fscores:\n', ["{:.3f}".format(f1) for f1 in fscores])
    print('supports:\n', supports)
    print('micro P, micro R, micro F1', "{:.3f}".format(precision), "{:.3f}".format(recall), "{:.3f}".format(f1))
    
    #pdb.set_trace()
    #print(mrr_total/sample_cnt)
        


def main():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--max_token_len', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--num_workers', type=int, default=5)

    parser.add_argument('--distance_norm_factor', type=float, default = 100)
    parser.add_argument('--spatial_dist_fill', type=float, default = 90000)
    parser.add_argument('--num_classes', type=int, default = 9)

    parser.add_argument('--with_type', default=False, action='store_true')
    parser.add_argument('--sep_between_neighbors', default=False, action='store_true')
    parser.add_argument('--no_spatial_distance',  default=False, action='store_true')

    parser.add_argument('--bert_option', type=str, default='bert-base')
    parser.add_argument('--prediction_save_dir', type=str, default=None)

    parser.add_argument('--checkpoint_path', type=str, default=None)

                        
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')


    # out_dir not None, and out_dir does not exist, then create out_dir
    if args.prediction_save_dir is not None and not os.path.isdir(args.prediction_save_dir):
        os.makedirs(args.prediction_save_dir)

    testing(args)

if __name__ == '__main__':

    main()

    