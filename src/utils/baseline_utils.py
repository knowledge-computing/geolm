# from transformers import BertModel, BertTokenizerFast
# from transformers import RobertaModel, RobertaTokenizer
from transformers import AutoModel, AutoTokenizer
# from transformers import LukeTokenizer, LukeModel


def get_baseline_model(model_name):

    if model_name == 'bert-base':
        tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        model = AutoModel.from_pretrained('bert-base-cased')
       
    elif model_name == 'bert-large':
        tokenizer = AutoTokenizer.from_pretrained("bert-large-cased")
        model = AutoModel.from_pretrained('bert-large-cased')
        
        # config = BertConfig(hidden_size = 1024, intermediate_size = 4096, num_attention_heads=16, num_hidden_layers=24)
    elif model_name == 'roberta-base':
        name_str = 'roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        model = AutoModel.from_pretrained(name_str)
        
    elif model_name == 'roberta-large':
        tokenizer = AutoTokenizer.from_pretrained('roberta-large')
        model = AutoModel.from_pretrained('roberta-large')
           
    elif model_name == 'spanbert-base':
        tokenizer = AutoTokenizer.from_pretrained('SpanBERT/spanbert-base-cased')
        model = AutoModel.from_pretrained('SpanBERT/spanbert-base-cased')
        
    elif model_name == 'spanbert-large':
        tokenizer = AutoTokenizer.from_pretrained('SpanBERT/spanbert-large-cased')
        model = AutoModel.from_pretrained('SpanBERT/spanbert-large-cased')
        

    elif model_name == 'simcse-bert-base':
        name_str = 'princeton-nlp/unsup-simcse-bert-base-uncased' # they don't have cased version for unsupervised
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        model = AutoModel.from_pretrained(name_str)
        
    elif model_name == 'simcse-bert-large':
        name_str = 'princeton-nlp/unsup-simcse-bert-large-uncased' # they don't have cased version for unsupervised
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        model = AutoModel.from_pretrained(name_str)
        

    elif model_name == 'simcse-roberta-base':
        name_str = 'rinceton-nlpp/unsup-simcse-roberta-base'
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        model = AutoModel.from_pretrained(name_str)
        
    elif model_name == 'simcse-roberta-large':
        name_str = 'princeton-nlp/unsup-simcse-roberta-large'
        tokenizer = AutoTokenizer.from_pretrained(name_str)
        model = AutoModel.from_pretrained(name_str)
        

    else:
        raise NotImplementedError


    return model, tokenizer # , config
