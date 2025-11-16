"""
Metrics calculation utilities for model evaluation.
"""

import numpy as np
import torch
from typing import Dict, Optional


def calculate_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        IoU score
    """
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    
    return float(intersection / union)


def calculate_dice(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate Dice coefficient.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        Dice score
    """
    intersection = np.logical_and(pred, target).sum()
    
    if pred.sum() + target.sum() == 0:
        return 1.0
    
    return float(2 * intersection / (pred.sum() + target.sum()))


def calculate_pixel_accuracy(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Calculate pixel-wise accuracy.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        Pixel accuracy
    """
    correct = (pred == target).sum()
    total = pred.size
    
    return float(correct / total)


def calculate_precision_recall_f1(
    pred: np.ndarray, 
    target: np.ndarray
) -> Dict[str, float]:
    """
    Calculate precision, recall, and F1 score.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        Dictionary with precision, recall, and F1
    """
    tp = np.logical_and(pred, target).sum()
    fp = np.logical_and(pred, ~target).sum()
    fn = np.logical_and(~pred, target).sum()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1)
    }


def calculate_all_metrics(
    pred: np.ndarray,
    target: np.ndarray
) -> Dict[str, float]:
    """
    Calculate all segmentation metrics.
    
    Args:
        pred: Predicted binary mask
        target: Ground truth binary mask
        
    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'iou': calculate_iou(pred, target),
        'dice': calculate_dice(pred, target),
        'pixel_accuracy': calculate_pixel_accuracy(pred, target)
    }
    
    metrics.update(calculate_precision_recall_f1(pred, target))
    
    return metrics


class MetricsTracker:
    """Track metrics across batches."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.metrics = {
            'iou': [],
            'dice': [],
            'precision': [],
            'recall': [],
            'f1': [],
            'pixel_accuracy': []
        }
    
    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Update metrics with new predictions.
        
        Args:
            pred: Predicted binary mask
            target: Ground truth binary mask
        """
        batch_metrics = calculate_all_metrics(pred, target)
        
        for key, value in batch_metrics.items():
            self.metrics[key].append(value)
    
    def get_average_metrics(self) -> Dict[str, float]:
        """
        Get average metrics across all batches.
        
        Returns:
            Dictionary with average metrics
        """
        avg_metrics = {}
        for key, values in self.metrics.items():
            if values:
                avg_metrics[key] = float(np.mean(values))
            else:
                avg_metrics[key] = 0.0
        
        return avg_metrics
    
    def get_summary(self) -> str:
        """
        Get formatted summary of metrics.
        
        Returns:
            Formatted string with metrics
        """
        avg_metrics = self.get_average_metrics()
        
        summary = "Metrics Summary:\n"
        summary += "-" * 40 + "\n"
        for key, value in avg_metrics.items():
            summary += f"{key:15s}: {value:.4f}\n"
        summary += "-" * 40
        
        return summary
