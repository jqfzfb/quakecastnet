"""Quantile metrics for forecasting multiple quantiles per time step."""

from typing import List

import torch

from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric

class QuantileLoss(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calculated as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        super().__init__(quantiles=quantiles, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor, **kwargs) -> torch.Tensor:

        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            masked_target = target
            masked_y_pred = y_pred[..., i]
            errors = masked_target - masked_y_pred
            loss_q = torch.max((q - 1) * errors, q * errors).unsqueeze(-1)
            losses.append(loss_q)
            
        losses = 2 * torch.cat(losses, dim=2)
        return losses

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            idx = self.quantiles.index(0.5)
            y_pred = y_pred[..., idx]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        return y_pred


class QuantileMaskLoss(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calculated as

    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    """

    def __init__(
        self,
        quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98],
        **kwargs,
    ):
        """
        Quantile loss

        Args:
            quantiles: quantiles for metric
        """
        super().__init__(quantiles=quantiles, **kwargs)

    def loss(self, y_pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, **kwargs) -> torch.Tensor:
        
        # calculate quantile loss
        losses = []
        for i, q in enumerate(self.quantiles):
            masked_target = target
            masked_y_pred = y_pred[..., i]
            errors = masked_target - masked_y_pred
            loss_q = torch.max((q - 1) * errors, q * errors).unsqueeze(-1)
            losses.append(loss_q)
            
        losses = 2 * torch.cat(losses, dim=2)

        mask = mask.unsqueeze(-1)
        losses = losses * mask.to(losses.dtype) 
        losses = losses * torch.ones_like(mask).sum().to(losses.dtype) / mask.sum().to(losses.dtype)
        
        return losses

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        if y_pred.ndim == 3:
            idx = self.quantiles.index(0.5)
            y_pred = y_pred[..., idx]
        return y_pred

    def to_quantiles(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a quantile prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: prediction quantiles
        """
        return y_pred


class KLAnnealingScheduler:
    def __init__(self, 
                 start_epoch: int = 0, 
                 end_epoch: int = 100, 
                 min_beta: float = 0.0, 
                 max_beta: float = 0.1, 
                 mode="linear",
                ):
        
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        self.min_beta = min_beta
        self.max_beta = max_beta
        self.mode = mode

    def __call__(self, epoch: int) -> float:
        if epoch < self.start_epoch:
            return self.min_beta
        elif epoch >= self.end_epoch:
            return self.max_beta

        progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)

        if self.mode == "linear":
            return self.min_beta + progress * (self.max_beta - self.min_beta)
        elif self.mode == "sigmoid":
            # sigmoid-shaped growth centered at mid-point
            import math
            scale = 12  # adjust sharpness of sigmoid
            s = 1 / (1 + math.exp(-scale * (progress - 0.5)))
            return self.min_beta + s * (self.max_beta - self.min_beta)
        else:
            raise ValueError("Invalid mode: choose 'linear' or 'sigmoid'")

class KLMSELoss(MultiHorizonMetric):
    def __init__(
        self,
        beta: float = 0.1,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.beta = beta
        self.beta_scheduler = KLAnnealingScheduler(start_epoch=0, end_epoch=50, max_beta=1.0)
        self.current_epoch = 0

    def to_prediction(self, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Convert network prediction into a point prediction.

        Args:
            y_pred: prediction output of network

        Returns:
            torch.Tensor: point prediction
        """
        return y_pred
    
    def update_epoch(self, epoch: int):
        self.current_epoch = epoch
        if self.beta_scheduler is not None:
            self.beta = self.beta_scheduler(epoch)
    
    def loss(self, y_pred: torch.Tensor, target: torch.Tensor, kl_loss: torch.Tensor) -> torch.Tensor:

        # kl + mse
        mse_loss = (y_pred - target) ** 2  # shape (batch, time, target)

        kl_loss_expanded = kl_loss[:, None, None]  # shape (batch, 1, 1)
        losses = mse_loss + self.beta * kl_loss_expanded

        return losses





        


