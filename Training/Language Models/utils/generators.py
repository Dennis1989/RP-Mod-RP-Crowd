import torch, torch.nn as nn, torch.nn.functional as F
import csv, random
import numpy as np

def simple_collate_fn(batch):
    out = {}
    for key in batch[0].keys():
        values = [b[key] for b in batch]
        if isinstance(values[0], torch.Tensor):
            values = torch.cat(values, dim=0)
        else:
            values = torch.as_tensor(values)
        out[key] = values
    if 'ids' in out:
        ids = out['ids']
        del out['ids']
        return out, None, {'ids': ids}
    else:
        return out


class TextDatasetFineTuning(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, key, values, mlm_probability=0., 
                 text='text', label='label', encoding="utf-8", include_mlm=False,**kwargs):
        self.raw_text = []
        self.labels = []
        self.include_mlm = include_mlm
        
        with open(path, encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row[key]) in values:
                    self.raw_text.append(row[text])
                    self.labels.append(int(float(row[label])))
                    
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.mlm_probability = mlm_probability
        
    def __mask_tokens(self, inputs, mlm_probability):
        special_tokens_mask = torch.tensor([self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)\
                               for val in inputs['labels'].tolist()], dtype=torch.bool)
        
        ## Get masked_indices
        probability_matrix = torch.full(inputs['labels'].shape, mlm_probability).masked_fill_(special_tokens_mask, value=0.)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs['labels'][~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs['labels'].shape, 0.8)).bool() & masked_indices
        inputs['input_ids'][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs['labels'].shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs['labels'].shape, dtype=torch.long)
        inputs['input_ids'][indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs
    
    def __getitem__(self, idx):
        text = self.raw_text[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(text, return_tensors="pt", **self.kwargs)
        if self.include_mlm:
            inputs['next_sentence_label'] = label
            inputs['labels'] = inputs['input_ids'].clone()

            if self.mlm_probability > 0.:
                inputs = self.__mask_tokens(inputs, self.mlm_probability)
            else:
                inputs['labels'] = inputs['labels'].fill_(-100)
        else:
            inputs['labels'] = label
        inputs['ids'] = random.randint(1., 9.)
        return inputs

    def __len__(self):
        return len(self.labels)

    
    
class TextDatasetPreTraining(torch.utils.data.Dataset):
    def __init__(self, path, tokenizer, key, values, mlm_probability=0., single_sent_probability=0.,
                 text='text', label='label', encoding="utf-8", **kwargs):
        self.raw_text_hate = []
        self.raw_text_no_hate = []
        with open(path, encoding=encoding) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row[key]) in values:
                    if float(row[label]) == 1.:
                        self.raw_text_hate.append(row[text])
                    elif float(row[label]) == 0.:
                        self.raw_text_no_hate.append(row[text])
                    else:
                        print('Failed to understand label: ', row[label])
                    
        self.tokenizer = tokenizer
        self.kwargs = kwargs
        self.mlm_probability = mlm_probability
        self.single_sent_probability = single_sent_probability / 2.
        self.shuffle()
        
    def shuffle(self):
        random.shuffle(self.raw_text_hate)
        random.shuffle(self.raw_text_no_hate)
        
    def __mask_tokens(self, inputs, mlm_probability):
        special_tokens_mask = torch.tensor([self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)\
                               for val in inputs['labels'].tolist()], dtype=torch.bool)
        
        ## Get masked_indices
        probability_matrix = torch.full(inputs['labels'].shape, mlm_probability).masked_fill_(special_tokens_mask, value=0.)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        inputs['labels'][~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(inputs['labels'].shape, 0.8)).bool() & masked_indices
        inputs['input_ids'][indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(inputs['labels'].shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), inputs['labels'].shape, dtype=torch.long)
        inputs['input_ids'][indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs
    
    def __getitem__(self, idx):
        hate_text = self.raw_text_hate[idx]
        no_hate_text = self.raw_text_no_hate[idx]
        r = random.random()
        
        if 0. <= r < (0.5 - self.single_sent_probability):
            inputs = self.tokenizer(hate_text, no_hate_text, return_tensors="pt", **self.kwargs)
            inputs['next_sentence_label'] = 1
            
        elif (0.5 - self.single_sent_probability) <= r < 0.5:
            inputs = self.tokenizer(hate_text, return_tensors="pt", **self.kwargs)
            inputs['next_sentence_label'] = 1
            
        elif 0.5 <= r < (1.0 - self.single_sent_probability):
            inputs = self.tokenizer(no_hate_text, hate_text, return_tensors="pt", **self.kwargs)
            inputs['next_sentence_label'] = 0
            
        else:
            inputs = self.tokenizer(no_hate_text, return_tensors="pt", **self.kwargs)
            inputs['next_sentence_label'] = 0
        
        inputs['labels'] = inputs['input_ids'].clone()
        
        if self.mlm_probability > 0.:
            inputs = self.__mask_tokens(inputs, self.mlm_probability)
        else:
            inputs['labels'] = inputs['labels'].fill_(-100)
        return inputs

    def __len__(self):
        return min(len(self.raw_text_hate), len(self.raw_text_no_hate))