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
    def __init__(self, num_samples=2000, distance_threshold=0.05, sin_thresh=0.1, eps=1e-6):
        super().__init__()
        self.num_samples = num_samples
        self.distance_threshold = distance_threshold
        self.sin_thresh = sin_thresh
        self.eps = eps

    def _sample_triplets(self, target):
        B, _, H, W = target.shape
        device = target.device
        # one set of matched triplets
        uA = torch.randint(0, W, (B, self.num_samples), device=device)
        vA = torch.randint(0, H, (B, self.num_samples), device=device)
        uB = torch.randint(0, W, (B, self.num_samples), device=device)
        vB = torch.randint(0, H, (B, self.num_samples), device=device)
        uC = torch.randint(0, W, (B, self.num_samples), device=device)
        vC = torch.randint(0, H, (B, self.num_samples), device=device)
        return (uA, vA, uB, vB, uC, vC)

    def _gather(self, d, u, v):
        B = d.shape[0]
        b = torch.arange(B, device=d.device).unsqueeze(1)
        return d[b, 0, v, u]  # (B,N)

    def _build_points(self, u, v, d, H, W):
        # canonical intrinsics-free pseudo camera (centered)
        cx = (W - 1) * 0.5
        cy = (H - 1) * 0.5
        u_n = (u.to(d.dtype) - cx) / float(W)
        v_n = (v.to(d.dtype) - cy) / float(H)

        x = d * u_n
        y = d * v_n
        z = d
        return torch.stack([x, y, z], dim=-1)  # (B,N,3)

    def get_normals(self, pred, target):
        B, _, H, W = target.shape
        uA, vA, uB, vB, uC, vC = self._sample_triplets(target)

        dA_gt = self._gather(target, uA, vA)
        dB_gt = self._gather(target, uB, vB)
        dC_gt = self._gather(target, uC, vC)

        dA_pr = self._gather(pred, uA, vA)
        dB_pr = self._gather(pred, uB, vB)
        dC_pr = self._gather(pred, uC, vC)

        # validity: require both gt and pred depths > 0
        valid = (dA_gt > 0) & (dB_gt > 0) & (dC_gt > 0) & (dA_pr > 0) & (dB_pr > 0) & (dC_pr > 0)
        if not valid.any():
            return None, None

        PA_gt = self._build_points(uA, vA, dA_gt, H, W)
        PB_gt = self._build_points(uB, vB, dB_gt, H, W)
        PC_gt = self._build_points(uC, vC, dC_gt, H, W)

        PA_pr = self._build_points(uA, vA, dA_pr, H, W)
        PB_pr = self._build_points(uB, vB, dB_pr, H, W)
        PC_pr = self._build_points(uC, vC, dC_pr, H, W)

        AB = PB_gt - PA_gt
        AC = PC_gt - PA_gt
        BC = PC_gt - PB_gt

        # R2: long-range distances on GT geometry
        distAB = torch.linalg.norm(AB, dim=-1)
        distAC = torch.linalg.norm(AC, dim=-1)
        distBC = torch.linalg.norm(BC, dim=-1)
        valid = valid & (distAB > self.distance_threshold) & (distAC > self.distance_threshold) & (distBC > self.distance_threshold)
        if not valid.any():
            return None, None

        # R1: non-collinear via sin(angle) (scale-invariant)
        cross = torch.cross(AB, AC, dim=-1)
        cross_norm = torch.linalg.norm(cross, dim=-1)
        sinang = cross_norm / (distAB * distAC).clamp_min(self.eps)
        valid = valid & (sinang > self.sin_thresh)
        if not valid.any():
            return None, None

        # normals (gt & pred)
        n_gt = torch.cross(PB_gt - PA_gt, PC_gt - PA_gt, dim=-1)
        n_pr = torch.cross(PB_pr - PA_pr, PC_pr - PA_pr, dim=-1)

        n_gt_norm = torch.linalg.norm(n_gt, dim=-1)
        n_pr_norm = torch.linalg.norm(n_pr, dim=-1)
        valid = valid & (n_gt_norm > self.eps) & (n_pr_norm > self.eps)
        if not valid.any():
            return None, None

        n_gt = F.normalize(n_gt[valid], dim=-1, eps=self.eps)
        n_pr = F.normalize(n_pr[valid], dim=-1, eps=self.eps)
        return n_pr, n_gt

    def forward(self, pred, target):
        normals = self.get_normals(pred, target)
        if normals[0] is None:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        n_pred, n_target = normals

        # sign-robust L1
        l_minus = (n_pred - n_target).abs().sum(dim=-1)
        l_plus  = (n_pred + n_target).abs().sum(dim=-1)
        loss = torch.minimum(l_minus, l_plus).mean()
        return loss