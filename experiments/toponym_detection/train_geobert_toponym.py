import os
import sys
from transformers import RobertaTokenizer, BertTokenizer, BertTokenizerFast
from tqdm import tqdm  # for our progress bar
from transformers import AdamW
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader

sys.path.append('../../src/')
from models.spatial_bert_model import SpatialBertModel
from models.spatial_bert_model import SpatialBertConfig
from models.spatial_bert_model import  SpatialBertForTokenClassification
from dataset_utils.lgl_data_loader import ToponymDataset
from pytorch_metric_learning import losses

import numpy as np
import argparse 
import pdb


DEBUG = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

use_amp = True # whether to use automatic mixed precision



def training(args):

    num_workers = args.num_workers
    batch_size = args.batch_size 
    epochs = args.epochs
    lr = args.lr 
    save_interval = args.save_interval
    max_token_len = args.max_token_len
    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill=args.spatial_dist_fill

    
    model_save_dir = os.path.join(args.model_save_dir, args.model_option)
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)
    
    print('model_save_dir', model_save_dir)
    print('\n')

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    if args.model_option == 'geobert-base':
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        config = SpatialBertConfig()
    elif args.model_option == 'geobert-large':
        tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")
        config = SpatialBertConfig( hidden_size = 1024, intermediate_size = 4096, num_attention_heads=16, num_hidden_layers=24)
    elif args.model_option == 'geobert-simcse-base':
        tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/unsup-simcse-bert-base-uncased') 
        config = SpatialBertConfig()
    else:
        raise NotImplementedError

    config.num_labels = 3
    # config.vocab_size = 28996 # for bert-cased 
    config.vocab_size = tokenizer.vocab_size

    model = SpatialBertForTokenClassification(config)

    model.load_state_dict(torch.load(args.model_checkpoint_path)['model'] , strict = False) 

    train_val_dataset = ToponymDataset(data_file_path = args.input_file_path, 
                                        tokenizer = tokenizer, 
                                        max_token_len = max_token_len, 
                                        distance_norm_factor = distance_norm_factor, 
                                        spatial_dist_fill = spatial_dist_fill, 
                                        mode = 'train'
                                        )

    percent_80 = int(len(train_val_dataset) * 0.8)
    train_dataset, val_dataset = torch.utils.data.random_split(train_val_dataset, [percent_80, len(train_val_dataset) - percent_80])

    # pdb.set_trace()
    train_loader = DataLoader(train_dataset, batch_size= batch_size, num_workers=num_workers,
                            shuffle=True, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, num_workers=num_workers,
                            shuffle=False, pin_memory=True, drop_last=False)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()


    # initialize optimizer
    optim = torch.optim.AdamW(model.parameters(), lr = lr)

    print('start training...')

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(train_loader, leave=True)
        iter = 0
        for batch in loop:

            with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = use_amp):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sent_position_ids = batch['sent_position_ids'].to(device)
                norm_lng_list = batch['norm_lng_list'].to(device)
                norm_lat_list = batch['norm_lat_list'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)

                outputs = model(
                    input_ids = input_ids, 
                    attention_mask = attention_mask, 
                    labels = labels, 
                    sent_position_ids = sent_position_ids,         
                    spatial_position_list_x = norm_lng_list, 
                    spatial_position_list_y = norm_lat_list,
                    token_type_ids = token_type_ids 
                    )

                loss = outputs.loss


                scaler.scale(loss).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()


                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix({'loss':loss.item()})


                iter += 1

                if iter % save_interval == 0 or iter == loop.total:
                    loss_valid = validating(val_loader, model, device)
                    print('validation loss', loss_valid)

                    save_path = os.path.join(model_save_dir, 'ep'+str(epoch) + '_iter'+ str(iter).zfill(5) \
                        + '_' +str("{:.4f}".format(loss.item())) + '_val' + str("{:.4f}".format(loss_valid)) +'.pth' )

                    checkpoint = {"model": model.state_dict(),
                        "optimizer": optim.state_dict(),
                        "scaler": scaler.state_dict()}
                    torch.save(checkpoint, save_path)

                    print('saving model checkpoint to', save_path)


def validating(val_loader, model, device):

    with torch.no_grad():

        loss_valid = 0
        loop = tqdm(val_loader, leave=True)
        data_count = 0
        for batch in loop:
            with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = use_amp):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                sent_position_ids = batch['sent_position_ids'].to(device)
                norm_lng_list = batch['norm_lng_list'].to(device)
                norm_lat_list = batch['norm_lat_list'].to(device)

                outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, sent_position_ids = sent_position_ids, spatial_position_list_x = norm_lng_list, 
                spatial_position_list_y = norm_lat_list)

                data_count += input_ids.shape[0]
                loss_valid  += outputs.loss * input_ids.shape[0]
                
        loss_valid = loss_valid / data_count

        return loss_valid

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--save_interval', type=int, default=2000)
    parser.add_argument('--max_token_len', type=int, default=512)
    

    parser.add_argument('--lr', type=float, default = 1e-5)
    parser.add_argument('--distance_norm_factor', type=float, default = 100)
    parser.add_argument('--spatial_dist_fill', type=float, default = 900)


    parser.add_argument('--model_option', type=str, default='geobert-base', choices=['geobert-base','geobert-large','geobert-simcse-base'])
    parser.add_argument('--model_checkpoint_path', type = str, default = None)
    parser.add_argument('--model_save_dir', type=str, default=None)

    parser.add_argument('--input_file_path', type=str, default='/home/zekun/toponym_detection/lgl/lgl.json')

    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    
    #  out_dir not None, and out_dir does not exist, then create out_dir
    if args.model_save_dir is not None and not os.path.isdir(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    training(args)


   
if __name__ == '__main__':

    main()