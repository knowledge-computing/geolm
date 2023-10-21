import os
import numpy as np 
import json 
import pdb 
#from sklearn.metrics.pairwise import cosine_similarity


import torch

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


def write_to_csv(out_dir, map_name, match_list):
    out_path = os.path.join(out_dir, map_name + '.json')

    with open(out_path, 'w') as f:
        for match_dict in match_list:
            json.dump(match_dict, f)
            f.write('\n')



def load_spatial_bert_pretrained_weights(model, weight_path):
   
    # load pretrained weights from SpatialBertLM to SpatialBertModel
    #pre_trained_model=torch.load(os.path.join(model_save_dir, weight_file_name))
    pre_trained_model=torch.load(weight_path)

    if 'model' in pre_trained_model:
        pre_trained_model = pre_trained_model["model"]

    cnt_layers = 0
    cur_model_kvpair=model.state_dict()
    for key,value in cur_model_kvpair.items():
        if 'bert.'+key in pre_trained_model:
            cur_model_kvpair[key]=pre_trained_model['bert.'+key]     
            #print("weights loaded for", key)
            cnt_layers += 1
        else:
            print("No weight for", key)

    print(cnt_layers, 'layers loaded')

    model.load_state_dict(cur_model_kvpair)
    
    return model



def get_spatialbert_embedding(entity, model, use_distance = True, agg = 'mean'):
    
    pseudo_sentence = entity['pseudo_sentence'][None,:].to(device)
    attention_mask = entity['attention_mask'][None,:].to(device)
    sent_position_ids = entity['sent_position_ids'][None,:].to(device)
    pivot_token_len = entity['pivot_token_len'] 

    
    if 'norm_lng_list' in entity and use_distance:
        position_list_x = entity['norm_lng_list'][None,:].to(device)
        position_list_y = entity['norm_lat_list'][None,:].to(device)
    else:
        position_list_x = []
        position_list_y = []

    outputs = model(input_ids = pseudo_sentence, attention_mask = attention_mask,  sent_position_ids = sent_position_ids,
        position_list_x = position_list_x, position_list_y = position_list_y)


    embeddings = outputs.last_hidden_state

    
    pivot_embed = embeddings[0][1:1+pivot_token_len]
    if agg == 'mean':
        pivot_embed = torch.mean(pivot_embed, axis = 0).detach().cpu().numpy() # (768, )
    elif agg == 'sum':
        pivot_embed = torch.sum(pivot_embed, axis = 0).detach().cpu().numpy() # (768, )
    else:
        raise NotImplementedError
    
    return pivot_embed

def get_bert_embedding(entity, model, agg = 'mean'):
    
    pseudo_sentence = entity['pseudo_sentence'].unsqueeze(0).to(device)
    attention_mask = entity['attention_mask'].unsqueeze(0).to(device)
    pivot_token_len = entity['pivot_token_len'] 
    
    
    outputs = model(input_ids = pseudo_sentence, attention_mask = attention_mask)


    embeddings = outputs.last_hidden_state

    
    pivot_embed = embeddings[0][1:1+pivot_token_len]
    if agg == 'mean':
        pivot_embed = torch.mean(pivot_embed, axis = 0).detach().cpu().numpy() # (768, )
    elif agg == 'sum':
        pivot_embed = torch.sum(pivot_embed, axis = 0).detach().cpu().numpy() # (768, )
    else:
        raise NotImplementedError
    
    
    return pivot_embed