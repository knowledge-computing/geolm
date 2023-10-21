import os
import sys
from transformers import RobertaTokenizer, BertTokenizer, BertTokenizerFast
from tqdm import tqdm  # for our progress bar
from transformers import AdamW
from transformers import AutoModel, AutoTokenizer
import torch
from torch.utils.data import DataLoader

sys.path.append('../../src/')

from dataset_utils.lgl_data_loader import ToponymDataset
from models.spatial_bert_model import SpatialBertConfig
from models.spatial_bert_model import  SpatialBertForTokenClassification
from pytorch_metric_learning import losses
import torchmetrics 

from seqeval.metrics import classification_report
import numpy as np
import argparse 
import pdb

DEBUG = False

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

use_amp = True # whether to use automatic mixed precision

id_to_label = {0:'O',1:'B-topo',2:'I-topo',-100:'O'}

def test(args):

    num_workers = args.num_workers
    batch_size = args.batch_size 
    max_token_len = args.max_token_len
    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill=args.spatial_dist_fill
    
    model_save_path = args.model_save_path
  


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
    # model.load_state_dict(bert_model.state_dict() , strict = False) # load sentence position embedding weights as well
    model.load_state_dict(torch.load(args.model_save_path)['model'], strict = True)
    
    test_dataset = ToponymDataset(data_file_path = args.input_file_path, 
                                        tokenizer = tokenizer, 
                                        max_token_len = max_token_len, 
                                        distance_norm_factor = distance_norm_factor, 
                                        spatial_dist_fill = spatial_dist_fill, 
                                        mode = 'test'
                                        )




    test_loader = DataLoader(test_dataset, batch_size= batch_size, num_workers=num_workers,
                            shuffle=False, pin_memory=True, drop_last=False)


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.eval()

    

    print('start testing...')

    all_labels = []
    all_pred = []
    precision_metric = torchmetrics.Precision(task='multiclass', num_classes = 3, average=None, ignore_index = -100).to(device)
    recall_metric = torchmetrics.Recall(task='multiclass', num_classes = 3, average=None, ignore_index = -100).to(device)
    f1_metric = torchmetrics.F1Score(task='multiclass', num_classes = 3, average=None, ignore_index = -100).to(device)

    # setup loop with TQDM and dataloader
    loop = tqdm(test_loader, leave=True)
    iter = 0
    count_1, count_2 = 0, 0
    for batch in loop:

        with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = use_amp):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            sent_position_ids = batch['sent_position_ids'].to(device)
            norm_lng_list = batch['norm_lng_list'].to(device)
            norm_lat_list = batch['norm_lat_list'].to(device)
            # pdb.set_trace()

            with torch.no_grad():
                outputs = model(input_ids = input_ids, attention_mask = attention_mask, labels = labels, sent_position_ids = sent_position_ids, spatial_position_list_x = norm_lng_list, 
                spatial_position_list_y = norm_lat_list)

            for i in range(input_ids.shape[0]):
                for l in labels[i]:
                    if l == 1:
                        count_1 += 1 
                    if l == 2:
                        count_2 += 1

            if DEBUG:
                for i in range(input_ids.shape[0]):
                    print(tokenizer.decode(input_ids[i]))
                    print(tokenizer.convert_ids_to_tokens(input_ids[i]))
                    print(labels[i])
                    print(torch.argmax(outputs.logits[i],axis=-1))
                    for l in labels[i]:
                        if l == 1:
                            count_1 += 1 
                        if l == 2:
                            count_2 += 1
                    for tmp1, tmp2 in zip(tokenizer.convert_ids_to_tokens(input_ids[i]), torch.argmax(outputs.logits[i],axis=-1)):
                        if tmp2 == 1 or tmp2== 2:
                            print(tmp1, tmp2)
                    pdb.set_trace()

            loss = outputs.loss
            logits = outputs.logits
            

            logits = torch.flatten(logits, start_dim = 0, end_dim = 1)
            labels = torch.flatten(labels, start_dim = 0, end_dim = 1)

            # pdb.set_trace()


            precision_metric(logits, labels)
            recall_metric(logits, labels)
            f1_metric(logits, labels)
            # pdb.set_trace()

            all_labels.append( [id_to_label[a] for a in labels.detach().cpu().numpy().tolist()])
            all_pred.append([id_to_label[a] for a in torch.argmax(logits,dim=1).detach().cpu().numpy().tolist()])

    print(count_1, count_2)
    total_precision = precision_metric.compute()
    total_recall = recall_metric.compute()
    total_f1 = f1_metric.compute()


    print(total_precision, total_recall, total_f1)
    print(classification_report(all_labels, all_pred, digits=5))




def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)

    parser.add_argument('--max_token_len', type=int, default=512)
    

    parser.add_argument('--distance_norm_factor', type=float, default = 100)
    parser.add_argument('--spatial_dist_fill', type=float, default = 90000)


    parser.add_argument('--model_option', type=str, default='geobert-base', choices=['geobert-base','geobert-large','geobert-simcse-base'])
    parser.add_argument('--model_save_path', type=str, default=None)

    parser.add_argument('--input_file_path', type=str, default='/home/zekun/toponym_detection/lgl/lgl.json')

   
    args = parser.parse_args()
    print('\n')
    print(args)
    print('\n')

    



    test(args)


   
if __name__ == '__main__':

    main()