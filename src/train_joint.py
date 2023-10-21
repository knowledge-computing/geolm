import os
import sys
from transformers import RobertaTokenizer, BertTokenizer, BertTokenizerFast
from tqdm import tqdm  # for our progress bar
from transformers import AdamW

import torch
from torch.utils.data import DataLoader, Sampler

sys.path.append('../../../')
from models.spatial_bert_model import SpatialBertModel
from models.spatial_bert_model import SpatialBertConfig
from models.spatial_bert_model import  SpatialBertForMaskedLM
from dataset_utils.osm_sample_loader import PbfMapDataset
from dataset_utils.paired_sample_loader import JointDataset
from transformers.models.bert.modeling_bert import BertForMaskedLM
from pytorch_metric_learning import losses
from transformers import AutoModel, AutoTokenizer

import numpy as np
import argparse 
import pdb


DEBUG = False
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

use_amp = True # whether to use automatic mixed precision


#TODO:
# unify pivot name and pivot len, ogc_fid in two loaders


class MyBatchSampler(Sampler):
    def __init__(self, batch_size, single_dataset_len):
        batch_size = batch_size // 2
        num_batches = single_dataset_len // batch_size
        batch_list = [] 
        for i in range(0, num_batches):
            cur_batch_list = []
            cur_batch_list.extend([j for j in range(i * batch_size, (i+1)*batch_size)]) 
            cur_batch_list.extend([j for j in range(i * batch_size + single_dataset_len, (i+1)*batch_size + single_dataset_len)]) 
            batch_list.append(cur_batch_list)

        self.batches = batch_list 
        # print(batch_list[0:100])

    def __iter__(self):
        for batch in self.batches:
            yield batch
    def __len__(self):
        return len(self.batches)

def training(args):

    num_workers = args.num_workers
    batch_size = args.batch_size //2
    epochs = args.epochs
    lr = args.lr #1e-7 # 5e-5
    save_interval = args.save_interval
    max_token_len = args.max_token_len
    distance_norm_factor = args.distance_norm_factor
    spatial_dist_fill=args.spatial_dist_fill
    # nl_dist_fill = args.nl_dist_fill
    with_type = args.with_type
    
    bert_option = args.bert_option
    if_no_spatial_distance = args.no_spatial_distance

    assert bert_option in ['bert-base','bert-large', 'simcse-base']

    
    model_save_dir = args.model_save_dir

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    
    print('model_save_dir', model_save_dir)
    print('\n')

    if bert_option == 'bert-base':
        # bert_model = BertForMaskedLM.from_pretrained('bert-base-cased')
        # tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

        # same as
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        bert_model = AutoModel.from_pretrained('bert-base-cased')

        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance)
    elif bert_option == 'simcse-base':
        name_str = 'princeton-nlp/unsup-simcse-bert-base-uncased' # they don't have cased version
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        bert_model = AutoModel.from_pretrained(name_str)

        # bert_model = BertForMaskedLM.from_pretrained('bert-base-cased')
        # tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance)

    elif bert_option == 'bert-large':

        # bert_model = BertForMaskedLM.from_pretrained('bert-large-cased')
        # tokenizer = BertTokenizerFast.from_pretrained("bert-large-cased")

        # same as 
        tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
        bert_model = AutoModel.from_pretrained('bert-large-cased')

        config = SpatialBertConfig(use_spatial_distance_embedding = not if_no_spatial_distance, hidden_size = 1024, intermediate_size = 4096, num_attention_heads=16, num_hidden_layers=24)
    else:
        raise NotImplementedError

    # config.vocab_size = 28996 # for bert-cased 
    config.vocab_size = tokenizer.vocab_size

    model = SpatialBertForMaskedLM(config)


    new_state_dict ={}
    # Modify the keys in the pretrained_dict to match the new model's prefix
    for key, value in bert_model.state_dict().items():

        new_key = key.replace("encoder.", "bert.encoder.")
        new_key = new_key.replace("embeddings.","bert.embeddings.")
        new_key = new_key.replace("word_bert.embeddings","word_embeddings")
        new_key = new_key.replace("position_bert.embeddings","position_embeddings")
        new_key = new_key.replace("token_type_bert.embeddings","token_type_embeddings")
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict, strict = False) # load sentence position embedding weights as well

    
    train_dataset_list = []
    continent_list = ['africa','antarctica','asia','australia_oceania',
                    'central_america','europe','north_america','south_america']


    geo_pseudo_sent_path = os.path.join(args.pseudo_sentence_dir,  'world.json')
    nl_sent_path = os.path.join(args.nl_sentence_dir,  'world.json')

    # take samples sequentially to gather hard negative in one batch
    train_dataset_hardneg = JointDataset(geo_file_path = geo_pseudo_sent_path,
                                nl_file_path = nl_sent_path, 
                                placename_to_osmid_path = args.placename_to_osmid_path,
                                tokenizer = tokenizer, 
                                max_token_len = max_token_len, 
                                distance_norm_factor = distance_norm_factor, 
                                spatial_dist_fill = spatial_dist_fill, 
                                sep_between_neighbors = True, 
                                label_encoder = None, 
                                if_rand_seq = False,
    )

    # take samples randomly
    train_dataset_randseq = JointDataset(geo_file_path = geo_pseudo_sent_path,
                                nl_file_path = nl_sent_path, 
                                placename_to_osmid_path = args.placename_to_osmid_path,
                                tokenizer = tokenizer, 
                                max_token_len = max_token_len, 
                                distance_norm_factor = distance_norm_factor, 
                                spatial_dist_fill = spatial_dist_fill, 
                                sep_between_neighbors = True, 
                                label_encoder = None,  
                                if_rand_seq = True           
    )

    train_dataset = torch.utils.data.ConcatDataset([train_dataset_hardneg, train_dataset_randseq])

    batch_sampler = MyBatchSampler(batch_size = batch_size, single_dataset_len = len(train_dataset_randseq)) 
    train_loader = DataLoader(train_dataset, num_workers=num_workers,  # batch_size= batch_size,
                                pin_memory=True, # drop_last=True,  shuffle=False, 
                                batch_sampler= batch_sampler) 
                                # shuffle needs to be false


    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)
    model.train()

    
    # initialize optimizer
    optim = torch.optim.AdamW(model.parameters(), lr = lr)

    contrastive_criterion = losses.NTXentLoss(temperature=0.01)

    if args.checkpoint_weight is not None:
        print('load weights from checkpoint', args.checkpoint_weight)
        checkpoint = torch.load(args.checkpoint_weight)
        model.load_state_dict(checkpoint["model"], strict = True)
        optim.load_state_dict(checkpoint["optimizer"])


    print('start training...')

    for epoch in range(epochs):
        # setup loop with TQDM and dataloader
        loop = tqdm(train_loader, leave=True)
        iter = 0
        for batch in loop:

            with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = use_amp):
                
                nl_data = batch['nl_data']
                geo_data = batch['geo_data']
                concat_data = batch['concat_data']
                
                nl_input_ids = nl_data['masked_input'].to(device)
                nl_entity_token_idx = nl_data['pivot_token_idx'].to(device)
                nl_attention_mask = nl_data['attention_mask'].to(device)
                nl_position_list_x = nl_data['norm_lng_list'].to(device)
                nl_position_list_y = nl_data['norm_lat_list'].to(device)
                nl_sent_position_ids = nl_data['sent_position_ids'].to(device)
                nl_pseudo_sentence = nl_data['pseudo_sentence'].to(device)
                nl_token_type_ids = nl_data['token_type_ids'].to(device)

                geo_input_ids = geo_data['masked_input'].to(device)
                geo_entity_token_idx = geo_data['pivot_token_idx'].to(device)
                geo_attention_mask = geo_data['attention_mask'].to(device)
                geo_position_list_x = geo_data['norm_lng_list'].to(device)
                geo_position_list_y = geo_data['norm_lat_list'].to(device)
                geo_sent_position_ids = geo_data['sent_position_ids'].to(device)
                geo_pseudo_sentence = geo_data['pseudo_sentence'].to(device)
                geo_token_type_ids = geo_data['token_type_ids'].to(device)

                joint_input_ids = concat_data['masked_input'].to(device)
                joint_attention_mask = concat_data['attention_mask'].to(device)
                joint_position_list_x = concat_data['norm_lng_list'].to(device)
                joint_position_list_y = concat_data['norm_lat_list'].to(device)
                joint_sent_position_ids = concat_data['sent_position_ids'].to(device)
                joint_pseudo_sentence = concat_data['pseudo_sentence'].to(device)
                joint_token_type_ids = concat_data['token_type_ids'].to(device)
                # pdb.set_trace()


                try:
                    outputs1 = model(joint_input_ids, attention_mask = joint_attention_mask, sent_position_ids = joint_sent_position_ids, pivot_token_idx_list=None,
                        spatial_position_list_x = joint_position_list_x, spatial_position_list_y = joint_position_list_y, token_type_ids = joint_token_type_ids, labels = joint_pseudo_sentence)
                except Exception as e:
                    print(e)
                    pdb.set_trace()

                # mlm for on joint geo and nl data 

                loss1 = outputs1.loss

                # loss1.backward()
                # optim.step()
                scaler.scale(loss1).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()


                outputs1 = model(geo_pseudo_sentence, attention_mask = geo_attention_mask, sent_position_ids = geo_sent_position_ids, pivot_token_idx_list=geo_entity_token_idx,
                    spatial_position_list_x = geo_position_list_x, spatial_position_list_y = geo_position_list_y, token_type_ids = geo_token_type_ids, labels = None)

                outputs2 = model(nl_pseudo_sentence, attention_mask = nl_attention_mask, sent_position_ids = nl_sent_position_ids,pivot_token_idx_list=nl_entity_token_idx,
                    spatial_position_list_x = nl_position_list_x, spatial_position_list_y = nl_position_list_y, token_type_ids = nl_token_type_ids, labels = None)

                embedding = torch.cat((outputs1.hidden_states, outputs2.hidden_states), 0)
                indicator = torch.arange(0, batch_size, dtype=torch.float32, requires_grad=False).to(device)
                indicator = torch.cat((indicator, indicator),0)
                loss3 = contrastive_criterion(embedding, indicator)

                if torch.isnan(loss3):
                    print(nl_entity_token_idx)
                    return 

                scaler.scale(loss3).backward()
                scaler.unscale_(optim)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

                loss = loss1 +  loss3
                # pdb.set_trace()

                loop.set_description(f'Epoch {epoch}')
                loop.set_postfix({'loss':loss.item(),'mlm':loss1.item(),'contrast':loss3.item()})


                if DEBUG:
                    print('ep'+str(epoch)+'_' + '_iter'+ str(iter).zfill(5), loss.item() )

                iter += 1

                if iter % save_interval == 0 or iter == loop.total:
                    save_path = os.path.join(model_save_dir, 'ep'+str(epoch) + '_iter'+ str(iter).zfill(5) \
                    + '_' +str("{:.4f}".format(loss.item())) +'.pth' )
                    checkpoint = {"model": model.state_dict(),
                        "optimizer": optim.state_dict(),
                        "scaler": scaler.state_dict()}
                    torch.save(checkpoint, save_path)
                    print('saving model checkpoint to', save_path)


def main():
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_workers', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--save_interval', type=int, default=8000)
    parser.add_argument('--max_token_len', type=int, default=512)

    parser.add_argument('--lr', type=float, default = 1e-5)
    parser.add_argument('--distance_norm_factor', type=float, default = 100)
    parser.add_argument('--spatial_dist_fill', type=float, default = 90000)
    # parser.add_argument('--nl_dist_fill', type=float, default = 0)

    parser.add_argument('--with_type', default=False, action='store_true')
    parser.add_argument('--no_spatial_distance',  default=False, action='store_true')

    parser.add_argument('--bert_option', type=str, default='bert-base', choices= ['bert-base','bert-large', 'simcse-base'])
    parser.add_argument('--model_save_dir', type=str, default='/data4/zekun/joint_model/weights_base_v1')
    parser.add_argument('--checkpoint_weight', type=str, default=None)

    parser.add_argument('--pseudo_sentence_dir', type=str, default = '/data4/zekun/osm_pseudo_sent/world_append_wikidata/')
    parser.add_argument('--nl_sentence_dir', type=str, default = '/data4/zekun/wikidata/world_georelation/joint_0618_valid')
    parser.add_argument('--placename_to_osmid_path', type=str, default = '/home/zekun/datasets/osm_pseudo_sent/name-osmid-dict/placename_to_osmid.json')
    
    # parser.add_argument('--wikidata_dir', type=str, default = '/data4/zekun/wikidata/world_geo/')
    #parser.add_argument('--wikipedia_dir', type=str, default = '/data2/wikipedia/separate_text_world/')
    #parser.add_argument('--trie_file_dir', type=str, default = '/data2/wikipedia/trie_output_world_format/')

    # CUDA_VISIBLE_DEVICES='2' python3 train_joint.py --model_save_dir='../../weights_base_run2' --pseudo_sentence_dir='../../datasets/osm_pseudo_sent/world_append_wikidata/' --nl_sentence_dir='../../datasets/wikidata/world_georelation/joint_v2/' --batch_size=32
    # CUDA_VISIBLE_DEVICES='3' python3 train_joint.py --bert_option='bert-large'  --pseudo_sentence_dir='../../datasets/osm_pseudo_sent/world_append_wikidata/' --nl_sentence_dir='../../datasets/wikidata/world_georelation/joint_v2/' --model_save_dir='../../weights_large' --batch_size=10 --lr=1e-6 
    
    # CUDA_VISIBLE_DEVICES='2' python3 train_joint.py --model_save_dir='../../weights_base_run2' --pseudo_sentence_dir='../../datasets/osm_pseudo_sent/world_append_wikidata/' --nl_sentence_dir='../../datasets/wikidata/world_georelation/joint_v2/' --batch_size=32 --checkpoint_weight='../../weights_base/ep1_iter60000_0.0440.pth'

    
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

    