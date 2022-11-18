import os, gpustat, random

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from nimble_pytorch import losses
from nimble_pytorch.losses import base, online_cma, transformers

from nimble_pytorch import metrics
from nimble_pytorch.metrics import accuracy, algorithm_selection, online_cma, transformers

BASE_METRICS = {
    'Base' : metrics.BaseMetric,
    'Entropy' : metrics.EntropyMetric,
    'Acc' : metrics.accuracy.Accuracy,
    'MacroAcc' : metrics.accuracy.MacroAccuracy,
    'MultiLabelAcc' : metrics.accuracy.MultiLabelAccuracy,
    'ASPerformance' : metrics.algorithm_selection.ASPerformance,
    'RelASPerformance' : metrics.algorithm_selection.ASPerformance,
    'CMA.rERT' : metrics.online_cma.RelativeERT,
    'CMA.Single_rERT' : metrics.online_cma.RelativeSingleERTMetric,
    'CMA.Advantage' : metrics.online_cma.AdvantageMetric,
    'CMA.Dynamic_rERT' : metrics.online_cma.DynamicRelativeERT,
    'CMA.Dynamic_rRT' : metrics.online_cma.DynamicRRT,
    'CMA.HLP_Acc' : metrics.online_cma.HLP_Acc,
    'Bert.Acc' : metrics.transformers.Accuracy,
    'Bert.AuC' : metrics.transformers.AreaUnderCurve,
}

BASE_LOSSES = {
    'L1Loss' : losses.L1Loss,
    'MSELoss' : losses.MSELoss,
    'CrossEntropyLoss' : losses.CrossEntropyLoss,
    'CTCLoss' : losses.CTCLoss,
    'NLLLoss' : losses.NLLLoss,
    'PoissonNLLLoss' : losses.PoissonNLLLoss,
    #'GaussianNLLLoss' : losses.GaussianNLLLoss,
    'KLDivLoss' : losses.KLDivLoss,
    'BCELoss' : losses.BCELoss,
    'BCEWithLogitsLoss' : losses.BCEWithLogitsLoss,
    'MarginRankingLoss' : losses.MarginRankingLoss,
    'HingeEmbeddingLoss' : losses.HingeEmbeddingLoss,
    'MultiLabelMarginLoss' : losses.MultiLabelMarginLoss, 
    #'HuberLoss' : losses.HuberLoss,
    'SmoothL1Loss' : losses.SmoothL1Loss,
    'SoftMarginLoss' : losses.SoftMarginLoss,
    'MultiLabelSoftMarginLoss' : losses.MultiLabelSoftMarginLoss,
    'CosineEmbeddingLoss' : losses.CosineEmbeddingLoss,
    'MultiMarginLoss' : losses.MultiMarginLoss,
    'TripletMarginLoss' : losses.TripletMarginLoss,
    'TripletMarginWithDistanceLoss' : losses.TripletMarginWithDistanceLoss,
    'LabelSmoothingCrossEntropy' : losses.base.LabelSmoothingCrossEntropy,
    'LabelSmoothingNLLLoss' : losses.base.LabelSmoothingNLLLoss,
    'WeightedMultiLabelCrossEntropyLoss' : losses.base.WeightedMultiLabelCrossEntropyLoss,
    'PenalizedNLLLoss' : losses.base.PenalizedNLLLoss,
    'BiLinearPenalizedNLLLoss' : losses.base.BiLinearPenalizedNLLLoss,
    'BiLinearPenalizedBCELoss' : losses.base.BiLinearPenalizedBCELoss,
    'BiLinearPenalizedBCELogitsLoss' : losses.base.BiLinearPenalizedBCELogitsLoss,
    'LinearPenalizedLoss' : losses.base.LinearPenalizedLoss,
    'LinearPenalizedLoss2' : losses.base.LinearPenalizedLoss2,
    'BilinearLoss' : losses.base.BilinearLoss,
    'CMA.RelativeERTLoss' : losses.online_cma.RelativeERTLoss,
    'CMA.RelativeSingleERTLoss' : losses.online_cma.RelativeSingleERTLoss,
    'CMA.ActorERTLoss' : losses.online_cma.ActorERTLoss,
    'CMA.ERTBaselineLoss' : losses.online_cma.ERTBaselineLoss,
    'CMA.rActorERTLoss' : losses.online_cma.RelativeActorERTLoss,
    'CMA.PopulationBaselineLoss' : losses.online_cma.PopulationBaselineLoss,
    'CMA.PopulationLoss' : losses.online_cma.PopulationLoss,
    'CMA.HLP_Loss' : losses.online_cma.HLP_Loss,
    'Transformer.HLoss' : losses.transformers.HLoss,
    'Transformer.NLLLoss' : losses.transformers.NLLLoss,
    'Transformer.DoubleLoss' : losses.transformers.DoubleLoss,
}

BASE_SCHEDULERS = {
    'LambdaLR' : lr_scheduler.LambdaLR,
    'MultiplicativeLR' : lr_scheduler.MultiplicativeLR,
    'StepLR' : lr_scheduler.StepLR,
    'MultiStepLR' : lr_scheduler.MultiStepLR,
    'ExponentialLR' : lr_scheduler.ExponentialLR,
    'CosineAnnealingLR' : lr_scheduler.CosineAnnealingLR,
    'ReduceLROnPlateau' : lr_scheduler.ReduceLROnPlateau,
    'CyclicLR' : lr_scheduler.CyclicLR,
    'OneCycleLR' : lr_scheduler.OneCycleLR,
    'CosineAnnealingWarmRestarts' : lr_scheduler.CosineAnnealingWarmRestarts,
}

BASE_OPTIMIZERS = {
    'Adadelta' : optim.Adadelta,
    'Adagrad' : optim.Adagrad,
    'Adam' : optim.Adam,
    'AdamW' : optim.AdamW,
    'SparseAdam' : optim.SparseAdam,
    'Adamax' : optim.Adamax,
    'ASGD' : optim.ASGD,
    'RMSprop' : optim.RMSprop,
    'Rprop' : optim.Rprop,
    'SGD' : optim.SGD,
}

def setGPU(ids=None, modes=['memory.free', 'temperature.gpu', 'fan.speed'], reset=False):
    if 'CUDA_DEVICE_SET' in os.environ and not reset:
        print(f"setGPU: GPU already set to: {os.environ['CUDA_VISIBLE_DEVICES']}")
        return
    
    stats = gpustat.GPUStatCollection.new_query()
    ids = list(range(len(stats.gpus))) if ids is None else ids
    ids = [ids] if not isinstance(ids, (tuple, list)) else ids
    stats = [st.entry for i, st in enumerate(stats) if i in ids]
    
    for gpu in stats:
        gpu['memory.ratio'] = float(gpu['memory.used']) / float(gpu['memory.total'])
        gpu['memory.free'] = float(gpu['memory.used']) - float(gpu['memory.total'])
    
    stats = sorted(stats, key = lambda gpu: [gpu[m] for m in modes])
    bestGPU = stats[0]['index']
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = str(bestGPU)
    os.environ['CUDA_DEVICE_SET'] = 'TRUE'
    print(f"setGPU: Setting GPU to: {bestGPU}")
    return