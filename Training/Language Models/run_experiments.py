import itertools, os
from multiprocessing import Pool
from time import sleep
from random import randint, shuffle
import subprocess, sys

BERT_TYPES = ['bert-base-german-cased', 
              'deepset/gbert-base', 
              'deepset/bert-base-german-cased-hatespeech-GermEval18Coarse']
DATASET_RP = ['RP-Crowd-2-folds', 'RP-Crowd-3-folds', 'RP-Mod-folds']
ALPHAS = ['0.1','0.5','0.9']

work_list_baseline_frozen = itertools.product(['train_models_baseline_frozen.py'], BERT_TYPES, DATASET_RP)
work_list_baseline_unfrozen = itertools.product(['train_models_baseline_unfrozen.py'], BERT_TYPES, DATASET_RP)
work_list_double_head = itertools.product(['train_models_double_head.py'], BERT_TYPES, DATASET_RP, ALPHAS)
work_list_der_standard = itertools.product(['train_models_double_head.py'], ['deepset/gbert-base'], 
                                           ['derstandard-folds'], ['0.5','0.9'])

work_list = [w for w in work_list_baseline_frozen] 
work_list += [w for w in work_list_baseline_unfrozen] 
work_list += [w for w in work_list_double_head]
work_list += [w for w in work_list_der_standard]
shuffle(work_list)

def execute(work):
    gpu, work_list = work[0], work[1]
    for args in work[1]:
        cmd = 'CUDA_VISIBLE_DEVICES={}, python3.9 '.format(gpu) + ' '.join(list(args))
        print(cmd)
        rtrn = os.popen(cmd).read()
        
split = len(work_list) // 3
work = []
work.append((0, work_list[:split]))
work.append((1, work_list[split:2*split]))
work.append((2, work_list[2*split:]))

print(*work, sep='\n')
#sys.exit()

with Pool(3) as p:
    _ = p.map(execute, work, chunksize=1)