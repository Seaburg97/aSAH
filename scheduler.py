# --coding:utf-8--
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

class WarmupReduceLROnPlateau(ReduceLROnPlateau):
    """Learning rate scheduler with warmup followed by ReduceLROnPlateau"""
    def __init__(self, optimizer, mode, warmup_steps, warmup_factor, patience, factor, verbose=True):
        super().__init__(optimizer, mode, patience=patience, factor=factor, verbose=verbose)
        self.warmup_steps = warmup_steps
        self.warmup_factor = warmup_factor
        self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=self.warmup_lambda)
        self.current_step = 0

    def warmup_lambda(self, step):
        return (self.warmup_factor * step / self.warmup_steps) + (1 - self.warmup_factor)

    def step(self, metrics):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            self.warmup_scheduler.step()
        else:
            super().step(metrics)

class WarmupCosineAnnealingLR(_LRScheduler):
    """Cosine annealing scheduler with linear warmup"""
    def __init__(self, optimizer: Optimizer, warmup_epochs: int, T_max: int, eta_min=0, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.T_max = T_max
        self.eta_min = eta_min
        self.cosine_scheduler = CosineAnnealingLR(optimizer, T_max, eta_min, last_epoch)
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            return [base_lr * (self.last_epoch + 1) / self.warmup_epochs for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            return self.cosine_scheduler.get_last_lr()

class WeightedMSELoss(nn.Module):
    """MSE loss with sample weighting based on class distribution"""
    def __init__(self, weight_type='balanced', pos_multiplier=2, threshold=2.5):
        super(WeightedMSELoss, self).__init__()
        self.weight_type = weight_type
        self.pos_multiplier = pos_multiplier
        self.threshold = threshold
        
    def forward(self, predictions, targets, sample_weights=None):
        loss = F.mse_loss(predictions, targets, reduction='none')
        
        if self.weight_type == 'balanced':
            # Calculate sample counts per class
            unique_targets = torch.unique(targets)
            target_counts = torch.tensor([torch.sum(targets == val) for val in unique_targets], 
                                       dtype=torch.float, 
                                       device=targets.device)
            
            # Calculate base weights
            weights = 1.0 / target_counts
            
            # Adjust positive sample weights
            pos_indices = unique_targets > self.threshold
            weights[pos_indices] = weights[pos_indices] * self.pos_multiplier
            
            # Normalize weights
            weights = weights / weights.sum()
            
            # Map weights to samples
            weight_map = {val.item(): weight.item() 
                         for val, weight in zip(unique_targets, weights)}
            sample_weights = torch.tensor([weight_map[target.item()] 
                                         for target in targets],
                                        device=predictions.device)
            
        weighted_loss = loss * sample_weights
        return torch.mean(weighted_loss)

def bmc_loss(pred, target, noise_var):
    """Bilateral-margin Classification Loss"""
    pred = pred.view(-1, 1)
    target = target.view(-1, 1)
    logits = - 0.5 * (pred - target.T).pow(2) / noise_var
    target_indices = torch.arange(pred.shape[0], device=pred.device)
    loss = F.cross_entropy(logits, target_indices)
    loss = loss * (2 * noise_var)
    return loss

class BMCLoss(nn.Module):
    """Bilateral-margin Classification Loss with learnable noise parameter"""
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))
        
    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)