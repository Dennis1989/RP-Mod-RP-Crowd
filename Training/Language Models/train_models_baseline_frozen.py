# coding: utf-8

import argparse, sys
sys.exit()
parser = argparse.ArgumentParser(description='Run Experiment')
parser.add_argument('bert_type', type=str)
parser.add_argument('dataset', type=str)
args = parser.parse_args()
print('train_models_baseline.py', args.bert_type, args.dataset)

from transformers import BertTokenizer, BertForSequenceClassification
from nimble_pytorch.transformers import NimbleBert
from utils.generators import TextDatasetFineTuning, simple_collate_fn
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pickle

from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

tokenizer = BertTokenizer.from_pretrained(args.bert_type)
model = BertForSequenceClassification.from_pretrained(args.bert_type)
def lr_lambda(epoch): return 0.97

train_d = TextDatasetFineTuning('../../Dataset/Text-Data/{}.csv'.format(args.dataset), tokenizer, 'ten_folds', 
                                [0,1,2,3,4,5,6,7], mlm_probability=0., padding='max_length', truncation=True, max_length=512)
valid_d = TextDatasetFineTuning('../../Dataset/Text-Data/{}.csv'.format(args.dataset), tokenizer, 'ten_folds', 
                                [8,9], mlm_probability=0., padding='max_length', truncation=True, max_length=512)

train_loader = DataLoader(train_d, batch_size=16, num_workers=8, shuffle=True, collate_fn=simple_collate_fn) #
valid_loader = DataLoader(valid_d, batch_size=16, num_workers=8, shuffle=False, collate_fn=simple_collate_fn)



model.classifier = nn.Sequential(
    nn.Linear(768, 384),
    nn.GELU(),
    nn.Linear(384, 2),
    nn.Softmax(-1)
)

model = NimbleBert(
        model,
        name = '{}-baseline-frozen-{}'.format(args.bert_type.split('/')[-1], args.dataset),
    ).cuda()

model.set_optimizer('AdamW', lr=1e-4, weight_decay=0.01)
model.set_scheduler('MultiplicativeLR', lr_lambda, last_epoch=-1)
model.set_loss('Transformer.HLoss')
model.add_metric('Bert.Acc', key_output='logits', key_input='labels')
model.add_metric('Bert.AuC', key_output='logits', key_input='labels', use_log=True)
model.freeze()

model.fit(train_loader, valid_loader, epochs=100, warm_up=256, auto_save_path='./trained_models', disabled_tqdm=True)


## Draw Acc
train_hates = np.array([np.array(model.train_history[epoch]['hate_acc']).mean() for epoch in model.train_history.keys()])
valid_hates = np.array([np.array(model.valid_history[epoch]['hate_acc']).mean() for epoch in model.valid_history.keys()])

fig = figure(figsize=(6, 5), dpi=128)
plt.title(model.name)
x = np.arange(len(train_hates))
plt.ylim([40, 100])
plt.ylabel('Acc', size='large')
plt.xlabel('Epoch', size='large')
plt.xticks(np.arange(len(train_hates)+1, step=max(len(train_hates)//10,1)))
plt.plot(x, train_hates)
plt.plot(x, valid_hates)
plt.hlines(train_hates.max(), 0, len(train_hates)-1, color='C0', linestyles='dotted')
plt.hlines(valid_hates.max(), 0, len(valid_hates)-1, color='C1', linestyles='dotted')
plt.text(10, 48, 'Acc.: {}'.format(train_hates.max().round(2)), color='C0')
plt.text(10, 45, 'Acc.: {}'.format(valid_hates.max().round(2)), color='C1')
fig.savefig('../../Evaluation/Figures/{}_acc.png'.format(model.name), dpi=fig.dpi)
del fig, train_hates, valid_hates

## Draw AuC
train_hates = np.array([np.array(model.train_history[epoch]['hate_auc']).mean() for epoch in model.train_history.keys()])
valid_hates = np.array([np.array(model.valid_history[epoch]['hate_auc']).mean() for epoch in model.valid_history.keys()])

fig = figure(figsize=(6, 5), dpi=128)
plt.title(model.name)
x = np.arange(len(train_hates))
plt.ylim([0.5, 1.])
plt.ylabel('AuC', size='large')
plt.xlabel('Epoch', size='large')
plt.xticks(np.arange(len(train_hates)+1, step=max(len(train_hates)//10,1)))
plt.plot(x, train_hates)
plt.plot(x, valid_hates)
plt.hlines(train_hates.max(), 0, len(train_hates)-1, color='C0', linestyles='dotted')
plt.hlines(valid_hates.max(), 0, len(valid_hates)-1, color='C1', linestyles='dotted')
plt.text(10, .48, 'AuC.: {}'.format(train_hates.max().round(2)), color='C0')
plt.text(10, .45, 'AuC.: {}'.format(valid_hates.max().round(2)), color='C1')
fig.savefig('../../Evaluation/Figures/{}_auc.png'.format(model.name), dpi=fig.dpi)
del fig, train_hates, valid_hates
NAME = model.name
del model

## Calc Predictions
model = torch.load('trained_models/{}_best.pt'.format(NAME)).cuda()
probs = []
decisions = []
y_trues = []
for inp in iter(valid_d):
    x = {}
    x['input_ids'] = inp['input_ids'].cuda()
    x['token_type_ids'] = inp['token_type_ids'].cuda()
    x['attention_mask'] = inp['attention_mask'].cuda()
    prob = model.predict(x)['logits'].cpu().exp()
    decision = prob.argmax(-1)
    probs.extend(prob.tolist())
    decisions.extend(decision.tolist())
    y_trues.append(inp['labels'])
    del x, prob, decision
del model


## Save Pickle
with open('../../Evaluation/Language Model-Results/{}_best.pickle'.format(NAME), 'wb') as f:
    pickle.dump({'probs': probs, 'decisions': decisions, 'y_trues': y_trues}, f)