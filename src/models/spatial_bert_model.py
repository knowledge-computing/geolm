import collections.abc
import math
import pdb

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss, MarginRankingLoss
# from typing import List, Optional, Tuple, Union

import transformers 
from transformers.activations import ACT2FN
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward, replace_return_docstrings
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, SequenceClassifierOutput
from transformers.modeling_utils import PreTrainedModel, find_pruneable_heads_and_indices, prune_linear_layer
from transformers.utils import logging

from transformers.modeling_utils import (
    PreTrainedModel,
    apply_chunking_to_forward,
    find_pruneable_heads_and_indices,
    prune_linear_layer,
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

# import def load_tf_weights_in_bert(model, config, tf_checkpoint_path):
# from .configuration_bert import BertConfig

from transformers.models.bert.modeling_bert import BertPooler, BertEncoder, BertLayer, BertOnlyMLMHead
from transformers.models.bert.modeling_bert import load_tf_weights_in_bert
from transformers.models.bert.modeling_bert import BertModel, BertPreTrainedModel


from transformers.configuration_utils import PretrainedConfig

class SpatialBertConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.BertModel` or a
    :class:`~transformers.TFBertModel`. It is used to instantiate a BERT model according to the specified arguments,
    defining the model architecture. Instantiating a configuration with the defaults will yield a similar configuration
    to that of the BERT `bert-base-uncased <https://huggingface.co/bert-base-uncased>`__ architecture.

    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.


    Args:
        vocab_size (:obj:`int`, `optional`, defaults to 30522):
            Vocabulary size of the BERT model. Defines the number of different tokens that can be represented by the
            :obj:`inputs_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        hidden_size (:obj:`int`, `optional`, defaults to 768):
            Dimensionality of the encoder layers and the pooler layer.
        num_hidden_layers (:obj:`int`, `optional`, defaults to 12):
            Number of hidden layers in the Transformer encoder.
        num_attention_heads (:obj:`int`, `optional`, defaults to 12):
            Number of attention heads for each attention layer in the Transformer encoder.
        intermediate_size (:obj:`int`, `optional`, defaults to 3072):
            Dimensionality of the "intermediate" (often named feed-forward) layer in the Transformer encoder.
        hidden_act (:obj:`str` or :obj:`Callable`, `optional`, defaults to :obj:`"gelu"`):
            The non-linear activation function (function or string) in the encoder and pooler. If string,
            :obj:`"gelu"`, :obj:`"relu"`, :obj:`"silu"` and :obj:`"gelu_new"` are supported.
        hidden_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout probability for all fully connected layers in the embeddings, encoder, and pooler.
        attention_probs_dropout_prob (:obj:`float`, `optional`, defaults to 0.1):
            The dropout ratio for the attention probabilities.
        max_position_embeddings (:obj:`int`, `optional`, defaults to 512):
            The maximum sequence length that this model might ever be used with. Typically set this to something large
            just in case (e.g., 512 or 1024 or 2048).
        type_vocab_size (:obj:`int`, `optional`, defaults to 2):
            The vocabulary size of the :obj:`token_type_ids` passed when calling :class:`~transformers.BertModel` or
            :class:`~transformers.TFBertModel`.
        initializer_range (:obj:`float`, `optional`, defaults to 0.02):
            The standard deviation of the truncated_normal_initializer for initializing all weight matrices.
        layer_norm_eps (:obj:`float`, `optional`, defaults to 1e-12):
            The epsilon used by the layer normalization layers.
        position_embedding_type (:obj:`str`, `optional`, defaults to :obj:`"absolute"`):
            Type of position embedding. Choose one of :obj:`"absolute"`, :obj:`"relative_key"`,
            :obj:`"relative_key_query"`. For positional embeddings use :obj:`"absolute"`. For more information on
            :obj:`"relative_key"`, please refer to `Self-Attention with Relative Position Representations (Shaw et al.)
            <https://arxiv.org/abs/1803.02155>`__. For more information on :obj:`"relative_key_query"`, please refer to
            `Method 4` in `Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)
            <https://arxiv.org/abs/2009.13658>`__.
        use_cache (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether or not the model should return the last key/values attentions (not used by all models). Only
            relevant if ``config.is_decoder=True``.
        classifier_dropout (:obj:`float`, `optional`):
            The dropout ratio for the classification head.

    Examples::

        >>> from transformers import BertModel, BertConfig

        >>> # Initializing a BERT bert-base-uncased style configuration
        >>> configuration = BertConfig()

        >>> # Initializing a model from the bert-base-uncased style configuration
        >>> model = BertModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        num_semantic_types=97,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        use_spatial_distance_embedding = True,
        classifier_dropout=None,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.use_spatial_distance_embedding = use_spatial_distance_embedding
        self.classifier_dropout = classifier_dropout
        self.num_semantic_types = num_semantic_types



class ContinuousSpatialPositionalEmbedding(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()

        self.emb_dim = int(hidden_size/2)  # dimension of the embedding

        inv_freq = 1 / (10000 ** (torch.arange(0.0, self.emb_dim) / self.emb_dim)) #(emb_dim)

        self.register_buffer("inv_freq", inv_freq)

    def forward(self, x ):
        bsz, seq_len = x.shape[0], x.shape[1] # get batch size
        flat_x = torch.flatten(x) # (bsize, seq_len) -> bsize * seq_len
        
        flat_sinusoid_inp = torch.ger(flat_x, self.inv_freq) # outer-product, out_shape: (bsize * seq_len, emb_dim)
        
        sinusoid_inp = flat_sinusoid_inp.reshape(bsz, seq_len, self.emb_dim) #(bsize * seq_len, emb_dim) -> (bsize, seq_len, emb_dim)
        
        ret_pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1) # (bsize, seq_len, 2*emb_dim)

        return ret_pos_emb



class SpatialBertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = SpatialBertConfig
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = 'spatialbert' 
    supports_gradient_checkpointing = True
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, BertEncoder):
            module.gradient_checkpointing = value


class SpatialEmbedding(nn.Module):
    # position_embedding_type controls the type for both sent_position_embedding and spatial_position_embedding

    def __init__(self, config):
        super().__init__()
        
        
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)

        #self.d_model = int(config.hidden_size/2) 

        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        
        self.sent_position_embedding = self.position_embeddings # a trick to simplify the weight loading from Bert

        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        

        self.sent_position_embedding_type = getattr(config, "position_embedding_type", "absolute")

        
        self.spatial_position_embedding = ContinuousSpatialPositionalEmbedding(hidden_size = config.hidden_size)
        self.spatial_position_embedding_type = getattr(config, "position_embedding_type", "absolute")


        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        
        
        self.use_spatial_distance_embedding = config.use_spatial_distance_embedding
        

    def forward(
        self,
        input_ids=None,
        token_type_ids = None, 
        sent_position_ids = None, 
        spatial_position_list_x=None,
        spatial_position_list_y = None,
    ):

        input_shape = input_ids.size()

        seq_length = input_shape[1]
        embeddings = self.word_embeddings(input_ids)
        #pdb.set_trace()
        
        if self.use_spatial_distance_embedding:
            if len(spatial_position_list_x) != 0 and len(spatial_position_list_y) !=0:
                if self.spatial_position_embedding_type == "absolute":
                    pos_emb_x = self.spatial_position_embedding(spatial_position_list_x)
                    pos_emb_y = self.spatial_position_embedding(spatial_position_list_y)

                    #print(embeddings.shape, pos_emb_x.shape, pos_emb_y.shape)
                    
                    embeddings +=  0.01* pos_emb_x
                    embeddings +=  0.01* pos_emb_y
                else:
                    raise NotImplementedError("Invalid spatial position embedding type")
                    # if relative, need to look at BertSelfAttention module as well

            else:
                pass
        else:
            pass


        if self.sent_position_embedding_type == "absolute":
            pos_emb_sent = self.sent_position_embedding(sent_position_ids)
            embeddings += pos_emb_sent
        else:
            raise NotImplementedError("Invalid sentence position embedding type")
            # if relative, need to look at BertSelfAttention module as well

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)
        
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings += token_type_embeddings 

        #pdb.set_trace()
        
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)



        return embeddings



class PivotEntityPooler(nn.Module):
    def __init__(self):
        super().__init__()
        
    # TODO: unify pivot_len_list and pivot_token_idx_list
    def forward(self, hidden_states, pivot_token_idx_list):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the tokens of target entity
        
        bsize = hidden_states.shape[0]

        #pdb.set_trace()

        tensor_list = []
        for i in torch.arange(0, bsize):
            # pivot_token_full = hidden_states[i, 1:pivot_len_list[i]+1]
            pivot_token_full = hidden_states[i, pivot_token_idx_list[i][0]:pivot_token_idx_list[i][1]]
            pivot_token_tensor = torch.mean(torch.unsqueeze(pivot_token_full, 0), dim = 1)
            tensor_list.append(pivot_token_tensor)


        batch_pivot_tensor = torch.cat(tensor_list, dim = 0)

        return batch_pivot_tensor
       


class SpatialBertModel(SpatialBertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well as a decoder, in which case a layer of
    cross-attention is added between the self-attention layers, following the architecture described in `Attention is
    all you need <https://arxiv.org/abs/1706.03762>`__ by Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit,
    Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the :obj:`is_decoder` argument of the configuration
    set to :obj:`True`. To be used in a Seq2Seq model, the model needs to initialized with both :obj:`is_decoder`
    argument and :obj:`add_cross_attention` set to :obj:`True`; an :obj:`encoder_hidden_states` is then expected as an
    input to the forward pass.
    """

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = SpatialEmbedding(config)
        self.encoder = BertEncoder(config)
        
        self.pivot_pooler = PivotEntityPooler() 

        # self.pooler = BertPooler(config) if add_pooling_layer else None
        

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} See base
        class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids = None, 
        sent_position_ids = None,
        spatial_position_list_x = None,
        spatial_position_list_y = None,
        pivot_token_idx_list = None,
        #pivot_len_list = None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention if
            the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask is used in
            the cross-attention if the model is configured as a decoder. Mask values selected in ``[0, 1]``:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))` of length :obj:`config.n_layers` with each tuple having 4 tensors of shape :obj:`(batch_size, num_heads, sequence_length - 1, embed_size_per_head)`):
            Contains precomputed key and value hidden states of the attention blocks. Can be used to speed up decoding.

            If :obj:`past_key_values` are used, the user can optionally input only the last :obj:`decoder_input_ids`
            (those that don't have their past key value states given to this model) of shape :obj:`(batch_size, 1)`
            instead of all :obj:`decoder_input_ids` of shape :obj:`(batch_size, sequence_length)`.
        use_cache (:obj:`bool`, `optional`):
            If set to :obj:`True`, :obj:`past_key_values` key value states are returned and can be used to speed up
            decoding (see :obj:`past_key_values`).
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        
        assert input_ids is not None
        input_shape = input_ids.size()

        
        
        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)



        embedding_output = self.embeddings(   
            input_ids=input_ids,
            token_type_ids=token_type_ids, 
            sent_position_ids = sent_position_ids,
            spatial_position_list_x= spatial_position_list_x,
            spatial_position_list_y = spatial_position_list_y,
        )
        

        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]
        if pivot_token_idx_list is not None:
            pooled_output = self.pivot_pooler(sequence_output, pivot_token_idx_list) 
        else:
            pooled_output = None

        # pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )




class SpatialBertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class SpatialBertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = SpatialBertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class SpatialBertOnlyMLMHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = SpatialBertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class SpatialBertOnlyTypingHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()
        
        self.seq_relationship = nn.Linear(config.hidden_size, config.num_semantic_types)

    def forward(self, pivot_pooled_output):

        pivot_pooled_output = self.dense(pivot_pooled_output)
        pivot_pooled_output = self.activation(pivot_pooled_output)

        seq_relationship_score = self.seq_relationship(pivot_pooled_output)
        return seq_relationship_score


class SpatialBertPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = SpatialBertLMPredictionHead(config)
        self.seq_relationship = nn.Linear(config.hidden_size, config.num_semantic_types)

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.predictions(sequence_output)
        seq_relationship_score = self.seq_relationship(pooled_output)
        return prediction_scores, seq_relationship_score


class SpatialBertForMaskedLM(SpatialBertPreTrainedModel):

    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `SpatialBertForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )

        self.bert = SpatialBertModel(config, add_pooling_layer=False)
        self.cls = SpatialBertOnlyMLMHead(config)
        self.pivot_pooler = PivotEntityPooler() 

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sent_position_ids = None,
        spatial_position_list_x = None,
        spatial_position_list_y = None,
        pivot_token_idx_list = None, 
        head_mask=None,
        encoder_attention_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        """

        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids, 
            sent_position_ids = sent_position_ids,
            spatial_position_list_x = spatial_position_list_x,
            spatial_position_list_y = spatial_position_list_y,
            head_mask=head_mask,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

      
        sequence_output = outputs[0]
        prediction_scores = self.cls(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()  # -100 index = padding token
            #print('loss_fct', print(prediction_scores.view(-1, self.config.vocab_size).shape, labels.view(-1).shape))
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))

        pooled_output = None
        if pivot_token_idx_list is not None:
            pooled_output = self.pivot_pooler(sequence_output, pivot_token_idx_list) 

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            pdb.set_trace()
            print('inside MLM', output.shape)
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return MaskedLMOutput(
            loss=masked_lm_loss,
            logits=prediction_scores,
            hidden_states=pooled_output,
            attentions=outputs.attentions,
        )


    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        effective_batch_size = input_shape[0]

        #  add a dummy token
        assert self.config.pad_token_id is not None, "The PAD token should be defined for generation"
        attention_mask = torch.cat([attention_mask, attention_mask.new_zeros((attention_mask.shape[0], 1))], dim=-1)
        dummy_token = torch.full(
            (effective_batch_size, 1), self.config.pad_token_id, dtype=torch.long, device=input_ids.device
        )
        input_ids = torch.cat([input_ids, dummy_token], dim=1)

        return {"input_ids": input_ids, "attention_mask": attention_mask}

class SpatialBertForSemanticTyping(SpatialBertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = SpatialBertModel(config)
        self.pivot_pooler = PivotEntityPooler() 
        self.num_semantic_types = config.num_semantic_types

        self.cls = SpatialBertOnlyTypingHead(config)

        self.init_weights()


    def forward(
        self,
        input_ids=None,
        token_type_ids=None,
        sent_position_ids = None,
        spatial_position_list_x = None,
        spatial_position_list_y = None,
        pivot_token_idx_list = None,
        attention_mask=None, 
        head_mask=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):


        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids = token_type_ids, 
            sent_position_ids = sent_position_ids,
            spatial_position_list_x = spatial_position_list_x,
            spatial_position_list_y = spatial_position_list_y,
            pivot_token_idx_list = pivot_token_idx_list,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = outputs[0]
        pooled_output = self.pivot_pooler(sequence_output, pivot_token_idx_list) 


        type_prediction_score = self.cls(pooled_output)

        typing_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            typing_loss = loss_fct(type_prediction_score.view(-1, self.num_semantic_types), labels.view(-1))

        if not return_dict:
            output = (type_prediction_score,) + outputs[2:]
            return ((typing_loss,) + output) if typing_loss is not None else output

        return SequenceClassifierOutput(
            loss=typing_loss,
            logits=type_prediction_score,
            #hidden_states=outputs.hidden_states,
            hidden_states = pooled_output,
            attentions=outputs.attentions,
        )

# this class does not have token type embedding
class SpatialBertForMarginRanking(SpatialBertPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        self.bert = SpatialBertModel(config)
        self.pivot_pooler = PivotEntityPooler() 

        self.init_weights()


    def forward(
        self,
        geo_entity_data,
        positive_type_data,
        negative_type_data,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        input_ids = geo_entity_data['pseudo_sentence'].to(device)
        attention_mask = geo_entity_data['attention_mask'].to(device)
        spatial_position_list_x = geo_entity_data['norm_lng_list'].to(device)
        spatial_position_list_y = geo_entity_data['norm_lat_list'].to(device)
        sent_position_ids = geo_entity_data['sent_position_ids'].to(device)

        # labels = batch['pivot_type'].to(device)
        pivot_lens = batch['pivot_token_len'].to(device)

        entity_outputs = model(input_ids, attention_mask = attention_mask, sent_position_ids = sent_position_ids,
            spatial_position_list_x = spatial_position_list_x, spatial_position_list_y = spatial_position_list_y).pooler_output

        
        input_ids = positive_type_data['pseudo_sentence'].to(device)
        attention_mask = positive_type_data['attention_mask'].to(device)
        spatial_position_list_x = positive_type_data['norm_lng_list'].to(device)
        spatial_position_list_y = positive_type_data['norm_lat_list'].to(device)
        sent_position_ids = positive_type_data['sent_position_ids'].to(device)

        positive_outputs = model(input_ids, attention_mask = attention_mask, sent_position_ids = sent_position_ids,
            spatial_position_list_x = spatial_position_list_x, spatial_position_list_y = spatial_position_list_y).pooler_output
        

        
        input_ids = negative_type_data['pseudo_sentence'].to(device)
        attention_mask = negative_type_data['attention_mask'].to(device)
        spatial_position_list_x = negative_type_data['norm_lng_list'].to(device)
        spatial_position_list_y = negative_type_data['norm_lat_list'].to(device)
        sent_position_ids = negative_type_data['sent_position_ids'].to(device)
        negative_outputs = model(input_ids, attention_mask = attention_mask, sent_position_ids = sent_position_ids,
            spatial_position_list_x = spatial_position_list_x, spatial_position_list_y = spatial_position_list_y).pooler_output
        



        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            sent_position_ids = sent_position_ids,
            spatial_position_list_x = spatial_position_list_x,
            spatial_position_list_y = spatial_position_list_y,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = outputs[0]
        pooled_output = self.pivot_pooler(sequence_output, pivot_len_list) 


        

        typing_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            typing_loss = loss_fct(type_prediction_score.view(-1, self.num_semantic_types), labels.view(-1))

        if not return_dict:
            output = (type_prediction_score,) + outputs[2:]
            return ((typing_loss,) + output) if typing_loss is not None else output

        return SequenceClassifierOutput(
            loss=typing_loss,
            logits=type_prediction_score,
            #hidden_states=outputs.hidden_states,
            hidden_states = pooled_output,
            attentions=outputs.attentions,
        )

class SpatialBertForTokenClassification(SpatialBertPreTrainedModel):

    # _keys_to_ignore_on_load_unexpected = [r"pooler"]

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = SpatialBertModel(config)
        
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        # Initialize weights and apply final processing
        # self.post_init()


    def forward(
        self,
        input_ids = None,
        token_type_ids=None,
        attention_mask = None,
        sent_position_ids = None,
        spatial_position_list_x = None,
        spatial_position_list_y = None,
        head_mask = None,
        labels = None,
        output_attentions = None,
        output_hidden_states = None,
        return_dict = None,
    ): # -> Union[Tuple[torch.Tensor], TokenClassifierOutput]:
        # r"""
        # labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
        #     Labels for computing the token classification loss. Indices should be in `[0, ..., config.num_labels - 1]`.
        # """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            token_type_ids= token_type_ids, 
            attention_mask=attention_mask,
            sent_position_ids = sent_position_ids,
            spatial_position_list_x = spatial_position_list_x,
            spatial_position_list_y = spatial_position_list_y,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )


        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
