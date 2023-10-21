import os
import sys
from transformers import RobertaTokenizer, BertTokenizer
from tqdm import tqdm  # for our progress bar
from transformers import AdamW
from transformers import BertTokenizerFast

import torch
from torch.utils.data import DataLoader

# sys.path.append('../../../')
sys.path.append('/home/zekun/joint_model/src')
from models.spatial_bert_model import SpatialBertModel
from models.spatial_bert_model import SpatialBertConfig
from models.spatial_bert_model import  SpatialBertForMaskedLM, SpatialBertForSemanticTyping
from datasets.osm_sample_loader import PbfMapDataset
from datasets.const import *
#from utils.common_utils import load_spatial_bert_pretrained_weights

from transformers.models.bert.modeling_bert import BertForMaskedLM

import numpy as np
import argparse 
from sklearn.preprocessing import LabelEncoder
import pdb


DEBUG = False


def training(args):

    num_workers = args.num_workers
    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr #1e-7 # 5e-5
    save_interval = args.save_interval
    max_token_len = args.max_token_len
    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill=args.spatial_dist_fill
    with_type = args.with_type
    sep_between_neighbors = args.sep_between_neighbors
    freeze_backbone = args.freeze_backbone
    mlm_checkpoint_path = args.mlm_checkpoint_path

    if_no_spatial_distance = args.no_spatial_distance


    bert_option = args.bert_option

    assert bert_option in ['bert-base','bert-large']

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


    if args.model_save_dir is None:
        checkpoint_basename = os.path.basename(mlm_checkpoint_path)
        checkpoint_prefix = checkpoint_basename.replace("mlm_mem_keeppos_","").strip('.pth')

        sep_pathstr = '_sep' if sep_between_neighbors else '_nosep' 
        freeze_pathstr = '_freeze' if freeze_backbone else '_nofreeze'
        if if_no_spatial_distance:
            model_save_dir = '/data3/zekun/spatial_bert_weights_ablation/'
        else:
            model_save_dir = '/data3/zekun/spatial_bert_weights/'
        model_save_dir = os.path.join(model_save_dir, 'typing_lr' + str("{:.0e}".format(lr)) + sep_pathstr +'_'+bert_option+ freeze_pathstr + '_london_california_bsize' + str(batch_size) )
        model_save_dir = os.path.join(model_save_dir, checkpoint_prefix)

        if not os.path.isdir(model_save_dir):
            os.makedirs(model_save_dir)
    else:
        model_save_dir = args.model_save_dir

    
    print('model_save_dir', model_save_dir)
    print('\n')

    if bert_option == 'bert-base':
        # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance, num_semantic_types=len(TYPE_LIST))
    elif bert_option == 'bert-large':
        # tokenizer = BertTokenizer.from_pretrained("bert-large-uncased")
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance, hidden_size = 1024, intermediate_size = 4096, num_attention_heads=16, num_hidden_layers=24, num_semantic_types=len(TYPE_LIST))
    else:
        raise NotImplementedError


    
    label_encoder = LabelEncoder()
    label_encoder.fit(TYPE_LIST)
   

    london_train_val_dataset = PbfMapDataset(data_file_path = london_file_path, 
                                        tokenizer = tokenizer, 
                                        max_token_len = max_token_len, 
                                        distance_norm_factor = distance_norm_factor, 
                                        spatial_dist_fill = spatial_dist_fill, 
                                        with_type = with_type,
                                        type_key_str = type_key_str,
                                        sep_between_neighbors = sep_between_neighbors, 
                                        label_encoder = label_encoder,
                                        mode = 'train')

    percent_80 = int(len(london_train_val_dataset) * 0.8)
    london_train_dataset, london_val_dataset = torch.utils.data.random_split(london_train_val_dataset, [percent_80, len(london_train_val_dataset) - percent_80])

    california_train_val_dataset = PbfMapDataset(data_file_path = california_file_path, 
                                            tokenizer = tokenizer, 
                                            max_token_len = max_token_len, 
                                            distance_norm_factor = distance_norm_factor, 
                                            spatial_dist_fill = spatial_dist_fill, 
                                            with_type = with_type,
                                            type_key_str = type_key_str,
                                            sep_between_neighbors = sep_between_neighbors,
                                            label_encoder = label_encoder,
                                            mode = 'train')
    percent_80 = int(len(california_train_val_dataset) * 0.8)
    california_train_dataset, california_val_dataset = torch.utils.data.random_split(california_train_val_dataset, [percent_80, len(california_train_val_dataset) - percent_80])

    train_dataset = torch.utils.data.ConcatDataset([london_train_dataset, california_train_dataset])
    val_dataset = torch.utils.data.ConcatDataset([london_val_dataset, california_val_dataset])


    if DEBUG:
        train_loader = DataLoader(train_dataset, batch_size= batch_size, num_workers=num_workers,
                                shuffle=False, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size= batch_size, num_workers=num_workers,
                                shuffle=False, pin_memory=True, drop_last=False)
    else:
        train_loader = DataLoader(train_dataset, batch_size= batch_size, num_workers=num_workers,
                                shuffle=True, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size= batch_size, num_workers=num_workers,
                                shuffle=False, pin_memory=True, drop_last=False)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    config.vocab_size = 28996 
    model = SpatialBertForSemanticTyping(config)
    model.to(device)

    #model = load_spatial_bert_pretrained_weights(model, mlm_checkpoint_path)

    # model.load_state_dict(torch.load(mlm_checkpoint_path), strict = False) # # load sentence position embedding weights as well
    model.load_state_dict(torch.load(mlm_checkpoint_path, map_location=torch.device(device))['model'], strict = False) 

    model.train()

    

    # initialize optimizer
    optim = AdamW(model.parameters(), lr = lr)

    print('start training...')

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(train_loader, leave=True)
        iter = 0
        for batch in loop:
            # initialize calculated gradients (from prev step)
            optim.zero_grad()
            # pull all tensor batches required for training
            input_ids = batch['pseudo_sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_list_x = batch['norm_lng_list'].to(device)
            position_list_y = batch['norm_lat_list'].to(device)
            sent_position_ids = batch['sent_position_ids'].to(device)

            #labels = batch['pseudo_sentence'].to(device)
            labels = batch['pivot_type'].to(device)
            entity_token_idx = batch['pivot_token_idx'].to(device)
            # pivot_lens = batch['pivot_token_len'].to(device)
            token_type_ids = torch.ones(input_ids.shape[0],max_token_len).int().to(device)

            # pdb.set_trace()

            outputs = model(input_ids, attention_mask = attention_mask, sent_position_ids = sent_position_ids,
                position_list_x = position_list_x, position_list_y = position_list_y, labels = labels, token_type_ids = token_type_ids,
                pivot_token_idx_list=entity_token_idx)


            loss = outputs.loss
            loss.backward()
            optim.step()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix({'loss':loss.item()})
            
            if DEBUG:
                print('ep'+str(epoch)+'_' + '_iter'+ str(iter).zfill(5), loss.item() )

            iter += 1

            if iter % save_interval == 0 or iter == loop.total:
                loss_valid = validating(val_loader, model, device)

                save_path = os.path.join(model_save_dir, 'keeppos_ep'+str(epoch) + '_iter'+ str(iter).zfill(5) \
                + '_' +str("{:.4f}".format(loss.item())) + '_val' + str("{:.4f}".format(loss_valid)) +'.pth' )

                torch.save(model.state_dict(), save_path)
                print('validation loss', loss_valid)
                print('saving model checkpoint to', save_path)

def validating(val_loader, model, device):

    with torch.no_grad():

        loss_valid = 0
        loop = tqdm(val_loader, leave=True)

        for batch in loop:
            input_ids = batch['pseudo_sentence'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            position_list_x = batch['norm_lng_list'].to(device)
            position_list_y = batch['norm_lat_list'].to(device)
            sent_position_ids = batch['sent_position_ids'].to(device)


            labels = batch['pivot_type'].to(device)
            entity_token_idx = batch['pivot_token_idx'].to(device)
            # pivot_lens = batch['pivot_token_len'].to(device)
            
            token_type_ids = torch.ones(input_ids.shape[0],512).int().to(device)

            outputs = model(input_ids, attention_mask = attention_mask, sent_position_ids = sent_position_ids,
                position_list_x = position_list_x, position_list_y = position_list_y, labels = labels, token_type_ids = token_type_ids,
                pivot_token_idx_list=entity_token_idx)

            loss_valid  += outputs.loss

        loss_valid /= len(val_loader)

        return loss_valid


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--max_token_len', type=int, default=512)
    

    parser.add_argument('--lr', type=float, default = 5e-5)
    parser.add_argument('--distance_norm_factor', type=float, default = 100)
    parser.add_argument('--spatial_dist_fill', type=float, default = 90000)
    parser.add_argument('--num_classes', type=int, default = 9)

    parser.add_argument('--with_type', default=False, action='store_true')
    parser.add_argument('--sep_between_neighbors', default=False, action='store_true')
    parser.add_argument('--freeze_backbone', default=False, action='store_true')
    parser.add_argument('--no_spatial_distance',  default=False, action='store_true')

    parser.add_argument('--bert_option', type=str, default='bert-base')
    parser.add_argument('--model_save_dir', type=str, default=None)

    parser.add_argument('--mlm_checkpoint_path', type=str, default=None)

                        
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')


    # out_dir not None, and out_dir does not exist, then create out_dir
    if args.model_save_dir is not None and not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    training(args)


if __name__ == '__main__':

    main()

    