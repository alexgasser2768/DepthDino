import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        mask = target > 0
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)
        
        diff = (target - pred).abs()

        with torch.no_grad():
            threshold = 0.2 * diff.max()

        l1_mask = mask & (diff <= threshold)
        l2_mask = mask & (diff > threshold)

        # BerHu loss
        loss = torch.zeros_like(diff, device=pred.device)
        loss[l1_mask] = diff[l1_mask]
        loss[l2_mask] = (diff[l2_mask] ** 2 + threshold ** 2) / (2 * threshold)

        return loss.mean()


class SILogLoss(nn.Module):
    def __init__(self, lambd=0.5, eps=1e-6):
        super().__init__()
        self.lambd = lambd
        self.eps = eps

    def forward(self, pred, target):
        # Only compute on valid depth values (avoid log(0) for target)
        mask = target > 0
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Clamp pred to avoid log(0) because model uses ReLU
        masked_pred = torch.clamp(pred[mask], min=self.eps) 
        
        # Log difference
        d = torch.log(masked_pred) - torch.log(target[mask])

        # Scale-invariant formula
        term1 = torch.mean(d ** 2)
        term2 = (torch.mean(d) ** 2)

        return torch.sqrt(term1 - self.lambd * term2)


class GradientMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def get_gradients(self, x):
        # Calculate gradients (finite differences)
        dx = x[:, :, :, 1:] - x[:, :, :, :-1]
        dy = x[:, :, 1:, :] - x[:, :, :-1, :]
        return dx, dy

    def forward(self, pred, target):
        mask = target > 0
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        pred_dx, pred_dy = self.get_gradients(pred)
        target_dx, target_dy = self.get_gradients(target)

        # Apply mask to gradients
        # (Note: mask needs to be sliced because gradients are 1px smaller)
        mask_x = mask[:, :, :, 1:] & mask[:, :, :, :-1]
        mask_y = mask[:, :, 1:, :] & mask[:, :, :-1, :]

        # Safety check for empty intersection
        if mask_x.sum() == 0 or mask_y.sum() == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        loss_x = F.l1_loss(pred_dx[mask_x], target_dx[mask_x])
        loss_y = F.l1_loss(pred_dy[mask_y], target_dy[mask_y])

        return loss_x + loss_y