import os
import sys
from transformers import RobertaTokenizer, BertTokenizer, BertTokenizerFast
from tqdm import tqdm  # for our progress bar
from transformers import AdamW
from transformers import AutoTokenizer, AutoModelForTokenClassification

import torch
from torch.utils.data import DataLoader

sys.path.append('../../src/')

from dataset_utils.lgl_data_loader import ToponymDataset
from transformers.models.bert.modeling_bert import BertForTokenClassification, BertModel
from transformers import BertConfig, RobertaForTokenClassification, AutoModelForTokenClassification
from pytorch_metric_learning import losses
from utils.baseline_utils import get_baseline_model

import numpy as np
import argparse 
import pdb


DEBUG = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

use_amp = True # whether to use automatic mixed precision


MODEL_OPTIONS = ['bert-base','bert-large','roberta-base','roberta-large',
'spanbert-base','spanbert-large','luke-base','luke-large',
'simcse-bert-base','simcse-bert-large','simcse-roberta-base','simcse-roberta-large',
'sapbert-base']


def training(args):

    num_workers = args.num_workers
    batch_size = args.batch_size 
    epochs = args.epochs
    lr = args.lr 
    save_interval = args.save_interval
    max_token_len = args.max_token_len
    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill=args.spatial_dist_fill


    backbone_option = args.backbone_option
    assert(backbone_option in MODEL_OPTIONS)

    
    model_save_dir = os.path.join(args.model_save_dir, backbone_option)
    if not os.path.isdir(model_save_dir):
        os.makedirs(model_save_dir)


    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    
    print('model_save_dir', model_save_dir)
    print('\n')

    # backbone_model, tokenizer = get_baseline_model(backbone_option)
    # config = backbone_model.config
    
    
    if backbone_option == 'bert-base':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModelForTokenClassification.from_pretrained("bert-base-cased", num_labels=3)

    elif backbone_option == 'roberta-base':
        tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        model = AutoModelForTokenClassification.from_pretrained("roberta-base", num_labels=3)

    elif backbone_option == 'spanbert-base':
        tokenizer = AutoTokenizer.from_pretrained('SpanBERT/spanbert-base-cased')
        model = AutoModelForTokenClassification.from_pretrained("SpanBERT/spanbert-base-cased", num_labels=3)
        # model.bert = backbone_model 

    elif backbone_option == 'sapbert-base':
        tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext") 
        model = AutoModelForTokenClassification.from_pretrained("cambridgeltl/SapBERT-from-PubMedBERT-fulltext", num_labels=3)
        
    elif backbone_option == 'simcse-bert-base':
        tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
        model = AutoModelForTokenClassification.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased", num_labels=3)
        
    else:
        raise NotImplementedError

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
                # pdb.set_trace()

                outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)

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
        data_count = 0
        loop = tqdm(val_loader, leave=True)

        for batch in loop:
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels)

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
    parser.add_argument('--distance_norm_factor', type=float, default = 100) # 0.0001)
    parser.add_argument('--spatial_dist_fill', type=float, default = 900) # 100)


    
    parser.add_argument('--backbone_option', type=str, default='bert-base')
    parser.add_argument('--model_save_dir', type=str, default=None)

    parser.add_argument('--input_file_path', type=str, default='/home/zekun/toponym_detection/lgl/lgl.json')

  
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