import torch
# add to train.py (top-level)

import torch.nn.functional as F
import logger

def feat_kd_loss(student_feat, teacher_feat):
    # simple L1 after spatial alignment
    if student_feat.shape[-2:] != teacher_feat.shape[-2:]:
        student_feat = F.interpolate(student_feat, size=teacher_feat.shape[-2:], mode="bilinear", align_corners=False)
    # channel mismatch? match by 1x1 projectors (see below) or use cosine loss with projections
    return F.l1_loss(student_feat, teacher_feat)



def si_log_kd_loss(student, teacher, mask, eps=1e-6):
    """
    Relative-depth KD:
      L = mean( | (log ds - mean) - (log dt - mean) | )
    This removes global scale (relative depth).
    """
    if not mask.any():
        return torch.tensor(0.0, device=student.device, requires_grad=True)

    s = torch.log(torch.clamp(student[mask], min=eps))
    t = torch.log(torch.clamp(teacher[mask], min=eps))

    s = s - s.mean()
    t = t - t.mean()
    return (s - t).abs().mean()


def train_one_epoch_distill(
    teacher, student, loader, optimizer,
    criterion_sup, device,
    alpha_kd=0.5,
    beta_feat=0.0,
    proj_s=None, proj_t=None
):
    teacher.eval()
    student.train()
    running = 0.0

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)

        with torch.no_grad():
            t_depth, t_feats = teacher(images, return_feats=True)

        s_out = student(images)
        if isinstance(s_out, tuple):
            s_depth, s_feats = s_out
        else:
            s_depth, s_feats = s_out, None

        # supervised loss (your current GT-based criterion)
        loss_sup = criterion_sup(s_depth, targets)

        # relative KD (scale-invariant) using valid GT pixels as mask
        valid = (targets > 0)
        loss_kd = si_log_kd_loss(s_depth, t_depth, valid)

        loss = loss_sup + alpha_kd * loss_kd

        # optional: feature KD on feat32 (deepest)
        if beta_feat > 0.0 and (s_feats is not None):
            s32 = s_feats[-1]
            t32 = t_feats[-1]
            if proj_s is not None and proj_t is not None:
                s32 = proj_s(s32)
                t32 = proj_t(t32)
            if s32.shape[-2:] != t32.shape[-2:]:
                s32 = F.interpolate(s32, size=t32.shape[-2:], mode="bilinear", align_corners=False)
            loss_feat = F.l1_loss(s32, t32)
            loss = loss + beta_feat * loss_feat

        if torch.isnan(loss):
            logger.warning("NaN loss detected!")
            continue

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running += loss.item()

    return running / len(loader)
