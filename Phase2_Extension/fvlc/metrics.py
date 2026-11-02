"""Evaluation metrics of the FVLC paper."""
import torch


@torch.no_grad()
def relative_change(a, b, reduction="per_sample"):
    """||a - b|| / ||a||.

    per_sample : mean over test samples of per-sample ratios (default, used in our tables)
    global     : Frobenius norm over the whole (N, k) matrix
    """
    if reduction == "global":
        return ((a - b).norm() / a.norm()).item()
    return ((a - b).norm(dim=1) / a.norm(dim=1).clamp_min(1e-12)).mean().item()


def tcpc(c_before, c_after, reduction="per_sample"):
    """Total Concept Perturbation Change, Sec. 6.1."""
    return relative_change(c_before, c_after, reduction)


def topc(y_before, y_after, reduction="per_sample"):
    """Total Output Perturbation Change, Sec. 6.1 (on the final-layer output vector)."""
    return relative_change(y_before, y_after, reduction)


@torch.no_grad()
def topk_overlap(a, b, k):
    """V_k(a, b) of Definition 1, averaged over samples."""
    ta = torch.topk(a, k, dim=1).indices
    tb = torch.topk(b, k, dim=1).indices
    inter = (ta.unsqueeze(2) == tb.unsqueeze(1)).any(dim=2).sum(dim=1).float()
    return (inter / k).mean().item()


@torch.no_grad()
def accuracy(logits, y):
    return (logits.argmax(dim=1).cpu() == y.cpu()).float().mean().item()


@torch.no_grad()
def kl_div(ref_logits, logits):
    """KL(softmax(ref) || softmax(logits)), averaged over samples."""
    p = torch.log_softmax(ref_logits, dim=1)
    q = torch.log_softmax(logits, dim=1)
    return (p.exp() * (p - q)).sum(dim=1).mean().item()
