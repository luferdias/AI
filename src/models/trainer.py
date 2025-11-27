"""
Training pipeline for crack detection models.
Supports U-Net, DeepLab, and YOLOv8 with MLflow tracking.
"""

import os
from typing import Optional, Dict, Any
from pathlib import Path
import logging

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
from tqdm import tqdm

logger = logging.getLogger(__name__)


class SegmentationTrainer:
    """Trainer for segmentation models (U-Net, DeepLab)."""
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        experiment_name: str = 'crack_detection'
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            device: Device to use for training
            experiment_name: MLflow experiment name
        """
        self.model = model.to(device)
        self.device = device
        self.experiment_name = experiment_name
        
        # Set up MLflow
        mlflow.set_experiment(experiment_name)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        loss_fn: Optional[nn.Module] = None,
        optimizer: Optional[optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        save_dir: Optional[Path] = None,
        log_interval: int = 10,
        early_stopping_patience: int = 10
    ) -> Dict[str, float]:
        """
        Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            learning_rate: Learning rate
            weight_decay: Weight decay for optimizer
            loss_fn: Loss function (defaults to BCEWithLogitsLoss)
            optimizer: Optimizer (defaults to Adam)
            scheduler: Learning rate scheduler
            save_dir: Directory to save model checkpoints
            log_interval: Logging interval
            early_stopping_patience: Patience for early stopping
            
        Returns:
            Dictionary with best metrics
        """
        # Default loss function - BCEWithLogits for binary segmentation
        if loss_fn is None:
            loss_fn = nn.BCEWithLogitsLoss()
        
        # Default optimizer
        if optimizer is None:
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        
        # Default scheduler
        if scheduler is None:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', factor=0.5, patience=5, verbose=True
            )
        
        # Setup save directory
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        # Start MLflow run
        with mlflow.start_run():
            # Log parameters
            mlflow.log_params({
                'epochs': epochs,
                'learning_rate': learning_rate,
                'weight_decay': weight_decay,
                'batch_size': train_loader.batch_size,
                'model_type': self.model.__class__.__name__
            })
            
            best_val_loss = float('inf')
            patience_counter = 0
            
            for epoch in range(epochs):
                # Training phase
                train_loss = self._train_epoch(
                    train_loader, optimizer, loss_fn, epoch, log_interval
                )
                
                # Validation phase
                val_loss, val_metrics = self._validate_epoch(
                    val_loader, loss_fn
                )
                
                # Update scheduler
                if scheduler:
                    scheduler.step(val_loss)
                
                # Log metrics
                mlflow.log_metrics({
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    **{f'val_{k}': v for k, v in val_metrics.items()}
                }, step=epoch)
                
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, "
                    f"Val IoU: {val_metrics.get('iou', 0):.4f}"
                )
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    
                    if save_dir:
                        checkpoint_path = save_dir / 'best_model.pth'
                        torch.save({
                            'epoch': epoch,
                            'model_state_dict': self.model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'val_loss': val_loss,
                            'val_metrics': val_metrics
                        }, checkpoint_path)
                        
                        # Log model to MLflow
                        mlflow.pytorch.log_model(self.model, "model")
                else:
                    patience_counter += 1
                
                # Early stopping
                if patience_counter >= early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break
            
            return {
                'best_val_loss': best_val_loss,
                'final_epoch': epoch
            }
    
    def _train_epoch(
        self,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        epoch: int,
        log_interval: int
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        
        pbar = tqdm(loader, desc=f'Training Epoch {epoch+1}')
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(self.device)
            masks = masks.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(images)
            loss = loss_fn(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % log_interval == 0:
                pbar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(loader)
    
    def _validate_epoch(
        self,
        loader: DataLoader,
        loss_fn: nn.Module
    ) -> tuple:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0
        total_iou = 0
        total_dice = 0
        
        with torch.no_grad():
            for images, masks in tqdm(loader, desc='Validating'):
                images = images.to(self.device)
                masks = masks.to(self.device)
                
                outputs = self.model(images)
                loss = loss_fn(outputs, masks)
                
                total_loss += loss.item()
                
                # Calculate metrics
                preds = torch.sigmoid(outputs) > 0.5
                iou = self._calculate_iou(preds, masks)
                dice = self._calculate_dice(preds, masks)
                
                total_iou += iou
                total_dice += dice
        
        metrics = {
            'iou': total_iou / len(loader),
            'dice': total_dice / len(loader)
        }
        
        return total_loss / len(loader), metrics
    
    @staticmethod
    def _calculate_iou(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Intersection over Union."""
        intersection = (pred & target).float().sum()
        union = (pred | target).float().sum()
        
        if union == 0:
            return 1.0
        
        return (intersection / union).item()
    
    @staticmethod
    def _calculate_dice(pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice coefficient."""
        intersection = (pred & target).float().sum()
        
        if pred.sum() + target.sum() == 0:
            return 1.0
        
        return (2 * intersection / (pred.sum() + target.sum())).item()
