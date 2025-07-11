import torch
import torch.nn.functional as F

def ppo_loss(old_logp, new_logp, adv, clip_eps=0.2):
    ratio = torch.exp(new_logp - old_logp)
    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
    return -torch.mean(torch.min(unclipped, clipped))

def grpo_loss(old_logp_group, new_logp_group, adv_group, clip_eps=0.2):
    """
    Implements Group Relative Policy Optimization:
    we compare each agent's logp not just to itself but to group average.
    For simplicity here we treat batch as one group.
    """
    avg_old = old_logp_group.mean()
    avg_new = new_logp_group.mean()
    delta_old = old_logp_group - avg_old
    delta_new = new_logp_group - avg_new
    ratio = torch.exp(delta_new - delta_old)
    unclipped = ratio * adv_group
    clipped = torch.clamp(ratio, 1-clip_eps, 1+clip_eps) * adv_group
    return -torch.mean(torch.min(unclipped, clipped))
