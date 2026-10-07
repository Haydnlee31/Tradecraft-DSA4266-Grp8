"""Opt-in loss-denominator control for the official-data study only.

Do not change historical losses. The default delegates to the existing factory.
The fixed denominator estimates the pooled weighted objective from a random batch;
it does NOT make multi-step local Adam plus FedAvg equivalent to central training.
"""
import torch
from torch import nn
from torch.nn import functional as F

from src.data.label_map import CLASSES
from src.models.losses import build_criterion


class FixedTrainingDenominatorCE(nn.Module):
    def __init__(self, weights, class_counts):
        super().__init__()
        counts = torch.tensor([class_counts[c] for c in CLASSES], dtype=torch.float64)
        if not torch.isfinite(counts).all() or torch.any(counts <= 0):
            raise ValueError('Finite positive training counts required')
        self.register_buffer('weight', weights.detach().clone())
        # Global expected target weight, computed ONLY from this training cohort.
        # Multiplying all class weights by a constant leaves this loss unchanged.
        expected = (counts * weights.double()).sum() / counts.sum()
        self.register_buffer('training_mean_weight', expected.to(weights.dtype))

    def forward(self, logits, target):
        terms = F.cross_entropy(logits, target, weight=self.weight, reduction='none')
        return terms.mean() / self.training_mean_weight


def build_official_criterion(loss_name, counts, reduction='batch_weight_sum'):
    if reduction not in ('batch_weight_sum', 'fixed_train_mean'):
        raise ValueError('Unknown official loss reduction')
    original = build_criterion(loss_name, counts)
    if reduction == 'batch_weight_sum':
        return original
    if loss_name not in ('sqrt_weighted_ce', 'weighted_ce'):
        raise ValueError('Fixed denominator control requires weighted cross entropy')
    return FixedTrainingDenominatorCE(original.weight, counts)
