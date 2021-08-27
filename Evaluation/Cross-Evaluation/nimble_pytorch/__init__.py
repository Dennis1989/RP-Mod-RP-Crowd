import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from nimble_pytorch import losses
from nimble_pytorch.losses import base, online_cma, transformers

from nimble_pytorch import metrics
from nimble_pytorch.metrics import accuracy, online_cma, transformers

BASE_METRICS = {
    'Base' : metrics.BaseMetric,
    'Acc' : metrics.accuracy.Accuracy,
    'CMA.rERT' : metrics.online_cma.RelativeERT,
    'CMA.Single_rERT' : metrics.online_cma.RelativeSingleERTMetric,
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
    'BilinearLoss' : losses.base.BilinearLoss,
    'CMA.RelativeERTLoss' : losses.online_cma.RelativeERTLoss,
    'CMA.RelativeSingleERTLoss' : losses.online_cma.RelativeSingleERTLoss,
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