import torch, torch.nn as nn, torch.nn.functional as F
from transformers.models.bert.modeling_bert import BertModel, BertPooler, BertPreTrainedModel, BertOnlyMLMHead, BertForSequenceClassification

class BertWrapper(nn.Module):
    def __init__(self, *args, **kwargs):
        super(BertWrapper, self).__init__()
        self.bert = BertForSequenceClassification.from_pretrained(*args, **kwargs)
        
    def forward(self, kwargs):
        return self.bert.forward(**kwargs)[0]
    
    def __call__(self, kwargs):
        return self.bert(**kwargs)[0]

class DoubleHeadedBert(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        
        hidden_output = (config.hidden_size + config.num_labels) // 2 + 1
        
        self.LMHead = BertOnlyMLMHead(config)
        self.pooler = BertPooler(config)
        self.dropout1 = nn.Dropout(config.hidden_dropout_prob)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)
        self.hidden = nn.Linear(config.hidden_size, hidden_output)
        self.classifier = nn.Linear(hidden_output, config.num_labels)
        
        self.init_weights()
        
    def freeze(self):
         for param in self.bert.parameters():
                param.requires_grad = False
                
    def unfreeze(self, last_froozen_layer_only=False):
        if not last_froozen_layer_only:
            params = self.bert
        else:
            all_params = [self.bert.embeddings] + list(self.bert.encoder.layer) ## List with all layers
            for params in all_params[::-1]: # Iterate reverse and find params where 
                if not all([p.requires_grad for p in params.parameters()]):
                    break
                    
        for param in params.parameters():
                param.requires_grad = True
        
    def forward(self, kwargs):
        input_kwargs = {
                'input_ids' : kwargs['input_ids'], 
                'token_type_ids' : kwargs['token_type_ids'] if 'token_type_ids' in kwargs else None, 
                'attention_mask': kwargs['attention_mask'] if 'attention_mask' in kwargs else None, 
                'position_ids': kwargs['position_ids'] if 'position_ids' in kwargs else None,
                'head_mask': kwargs['head_mask'] if 'head_mask' in kwargs else None,
                'inputs_embeds': kwargs['inputs_embeds'] if 'inputs_embeds' in kwargs else None,
                'encoder_hidden_states': kwargs['encoder_hidden_states'] if 'encoder_hidden_states' in kwargs else None,
                'encoder_attention_mask': kwargs['encoder_attention_mask'] if 'encoder_attention_mask' in kwargs else None,
                'output_attentions': kwargs['output_attentions'] if 'output_attentions' in kwargs else None,
                'output_hidden_states': kwargs['output_hidden_states'] if 'output_hidden_states' in kwargs else None,
                'return_dict': kwargs['return_dict'] if 'return_dict' in kwargs else False,
            }
        
        output = self.bert(**input_kwargs)[0]
        #print(output)
        
        cls_head = self.pooler(output)
        cls_head = F.relu(self.hidden(self.dropout1(cls_head)))
        cls_head = F.log_softmax(self.classifier(self.dropout2(cls_head)), dim=-1)
        
        if 'return_lm' in kwargs and kwargs['return_lm']:
            lm_head = F.log_softmax(self.LMHead(output), dim=-1)
            return (cls_head, lm_head)
        else:
            return cls_head

class RawPooledBert(BertPreTrainedModel):
    _keys_to_ignore_on_load_unexpected = [r"pooler"]
    _keys_to_ignore_on_load_missing = [r"position_ids", r"predictions.decoder.bias"]
    
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.bert = BertModel(config, add_pooling_layer=False)
        
        self.pooler = BertPooler(config)        
        
    def forward(self, kwargs):
        self.eval()
        input_kwargs = {
                'input_ids' : kwargs['input_ids'], 
                'token_type_ids' : kwargs['token_type_ids'] if 'token_type_ids' in kwargs else None, 
                'attention_mask': kwargs['attention_mask'] if 'attention_mask' in kwargs else None, 
                'position_ids': kwargs['position_ids'] if 'position_ids' in kwargs else None,
                'head_mask': kwargs['head_mask'] if 'head_mask' in kwargs else None,
                'inputs_embeds': kwargs['inputs_embeds'] if 'inputs_embeds' in kwargs else None,
                'encoder_hidden_states': kwargs['encoder_hidden_states'] if 'encoder_hidden_states' in kwargs else None,
                'encoder_attention_mask': kwargs['encoder_attention_mask'] if 'encoder_attention_mask' in kwargs else None,
                'output_attentions': kwargs['output_attentions'] if 'output_attentions' in kwargs else None,
                'output_hidden_states': kwargs['output_hidden_states'] if 'output_hidden_states' in kwargs else None,
                'return_dict': kwargs['return_dict'] if 'return_dict' in kwargs else False,
            }
        
        with torch.no_grad():
            output = self.bert(**input_kwargs)[0]
            return self.pooler(output)