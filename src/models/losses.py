"""
Loss functions for crack detection.
Includes specialized losses optimized for high recall.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice loss for segmentation.
    Better for imbalanced datasets.
    """
    
    def __init__(self, smooth: float = 1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss.
        
        Args:
            pred: Predicted logits
            target: Ground truth masks
            
        Returns:
            Dice loss value
        """
        pred = torch.sigmoid(pred)
        
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)
        
        return 1 - dice


class FocalLoss(nn.Module):
    """
    Focal loss for addressing class imbalance.
    Focuses on hard examples.
    """
    
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Focal loss.
        
        Args:
            pred: Predicted logits
            target: Ground truth
            
        Returns:
            Focal loss value
        """
        bce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """
    Combination of BCE, Dice, and Focal losses.
    Optimized for high recall in crack detection.
    """
    
    def __init__(
        self,
        bce_weight: float = 0.5,
        dice_weight: float = 0.3,
        focal_weight: float = 0.2
    ):
        super(CombinedLoss, self).__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.focal_weight = focal_weight
        
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.focal = FocalLoss()
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate combined loss.
        
        Args:
            pred: Predicted logits
            target: Ground truth masks
            
        Returns:
            Combined loss value
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)
        focal_loss = self.focal(pred, target)
        
        total_loss = (
            self.bce_weight * bce_loss +
            self.dice_weight * dice_loss +
            self.focal_weight * focal_loss
        )
        
        return total_loss


class RecallOptimizedLoss(nn.Module):
    """
    Loss optimized for high recall.
    Penalizes false negatives more heavily.
    """
    
    def __init__(self, beta: float = 2.0):
        """
        Initialize loss.
        
        Args:
            beta: Weight for recall (beta=2 means recall is 2x more important)
        """
        super(RecallOptimizedLoss, self).__init__()
        self.beta = beta
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate F-beta loss (optimized for recall).
        
        Args:
            pred: Predicted logits
            target: Ground truth
            
        Returns:
            F-beta loss
        """
        pred = torch.sigmoid(pred)
        
        # Calculate true positives, false positives, false negatives
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        fn = ((1 - pred) * target).sum()
        
        # F-beta score
        beta_squared = self.beta ** 2
        fbeta = ((1 + beta_squared) * tp) / (
            (1 + beta_squared) * tp + beta_squared * fn + fp + 1e-8
        )
        
        return 1 - fbeta
