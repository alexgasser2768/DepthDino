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


class AleatoricSurfaceNormalLoss(nn.Module):
    """
    Implements the Aleatoric Surface Normal Loss (Angular vonMF) from:
    'Estimating and Exploiting the Aleatoric Uncertainty in Surface Normal Estimation' [Bae et al. 2021]
    
    Ref: https://arxiv.org/abs/2109.09881 (Equation 5)
    """
    def __init__(self, use_intrinsics=False):
        super().__init__()
        self.use_intrinsics = use_intrinsics
        self.eps = 1e-6

    def depth_to_normal(self, depth):
        """
        Converts a depth map to surface normals using finite differences.
        Assumes depth is (B, 1, H, W).
        """
        # Calculate gradients (dy, dx)
        # Pad to maintain shape matching the input depth
        padded_depth = F.pad(depth, (0, 1, 0, 1), mode='replicate')

        # d_depth / dx
        dz_dx = padded_depth[:, :, :, 1:] - padded_depth[:, :, :, :-1]
        dz_dx = dz_dx[:, :, :depth.shape[2], :] # Crop back to H

        # d_depth / dy
        dz_dy = padded_depth[:, :, 1:, :] - padded_depth[:, :, :-1, :]
        dz_dy = dz_dy[:, :, :, :depth.shape[3]] # Crop back to W

        # Construct surface normals: [-dz/dx, -dz/dy, 1]
        # Note: Without explicit intrinsics, this is an approximation in image space.
        # This is standard for depth-consistency losses where intrinsics are fixed/unknown.
        normal = torch.cat([-dz_dx, -dz_dy, torch.ones_like(depth)], dim=1)

        # Normalize to unit vectors
        return F.normalize(normal, p=2, dim=1, eps=self.eps)

    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) Depth map [in meters]
            target: (B, 1, H, W) Ground Truth Depth map [in meters]
        """
        mask = target > 0
        if not mask.any():
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        # Convert Depth to Surface Normals
        pred_norm = self.depth_to_normal(pred)
        target_norm = self.depth_to_normal(target)

        # Compute Angular Error (Theta)
        # Dot product clamped to [-1, 1] for numerical stability of acos
        dot_product = torch.sum(pred_norm * target_norm, dim=1, keepdim=True)
        dot_product = torch.clamp(dot_product, -1.0 + self.eps, 1.0 - self.eps)
        theta = torch.acos(dot_product)

        # --- Angular Loss Only (Geometric Consistency) ---
        # Paper argues minimizing Angle is better than L2. 
        # This is effectively Eq. 5 with constant Kappa.
        return theta[mask].mean()


class VirtualNormalLoss(nn.Module):
    def __init__(self, num_samples=2000, distance_threshold=0.1):
        super().__init__()
        self.num_samples = num_samples
        self.distance_threshold = distance_threshold

    def get_points(self, pred, target):
        n, _, w, h = target.shape  # n is batch size

        # get (n x m x 1) in range [0, x] and [0, y] random integers
        u = torch.randint(0, w, (n, self.num_samples, 1), device=target.device)
        v = torch.randint(0, h, (n, self.num_samples, 1), device=target.device)
        
        # Sample points from z matrix (n x 1 x w x h) -> (n x m x 1)
        batch_indices = torch.arange(n, device=target.device).unsqueeze(1)

        pred_depths = pred[batch_indices, :, u.squeeze(-1), v.squeeze(-1)]
        target_depths = target[batch_indices, :, u.squeeze(-1), v.squeeze(-1)]

        # Create u, v, z matrices (n x m x 3)
        pred_output = torch.cat([u.float(), v.float(), pred_depths], dim=-1)
        target_output = torch.cat([u.float(), v.float(), target_depths], dim=-1)

        # Normalize coordinates by width and height
        pred_output[..., 0] /= w
        pred_output[..., 1] /= h
        target_output[..., 0] /= w
        target_output[..., 1] /= h

        return pred_output, target_output

    def get_normals(self, pred, target):
        p1_pred, p1_target = self.get_points(pred, target)
        p2_pred, p2_target = self.get_points(pred, target)
        p3_pred, p3_target = self.get_points(pred, target)

        # Check if points are too close or collinear and remove with a mask
        vec12_target = p2_target - p1_target
        vec13_target = p3_target - p1_target

        cross_target = torch.cross(vec12_target, vec13_target, dim=-1)
        norm_target = torch.norm(cross_target, dim=-1)

        # Remove points that are too close, collinear, or outliers (z = 0)
        mask = (norm_target > self.distance_threshold) & \
               (p1_target[..., 2] > 0) & \
               (p2_target[..., 2] > 0) & \
               (p3_target[..., 2] > 0)

        # Check if points are too close or collinear and remove with a mask
        vec12_pred = p2_pred - p1_pred
        vec13_pred = p3_pred - p1_pred

        cross_pred = torch.cross(vec12_pred, vec13_pred, dim=-1)

        # Return valid normal vectors for target and prediction
        n_target = F.normalize(cross_target[mask], dim=-1)
        n_pred = F.normalize(cross_pred[mask], dim=-1)
        
        return n_pred, n_target

    def forward(self, pred, target):
        n_pred, n_target = self.get_normals(pred, target)

        if n_pred.shape[0] == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return F.l1_loss(n_pred, n_target)
