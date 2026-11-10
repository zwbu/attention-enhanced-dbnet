import torch
from torch.nn import functional as F


def db_loss(outputs, targets, negative_ratio, threshold_weight, binary_weight):
    logits = outputs["probability_logits"][:, 0]
    target = targets["shrink"]
    valid = targets["valid"].bool()
    losses = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability_losses = []
    for loss_map, text, mask in zip(losses, target.bool(), valid):
        positive = loss_map[text & mask]
        negative = loss_map[~text & mask]
        count = min(negative.numel(), negative_ratio * max(1, positive.numel()))
        selected = negative.topk(count).values
        probability_losses.append((positive.sum() + selected.sum()) /
                                  max(1, positive.numel() + count))
    probability_loss = torch.stack(probability_losses).mean()
    border_mask = targets["border_mask"]
    threshold_loss = ((outputs["threshold"][:, 0] - targets["border"]).abs()
                      * border_mask).sum() / border_mask.sum().clamp_min(1)
    binary = outputs["binary"][:, 0] * valid
    text = target * valid
    intersection = (binary * text).sum(dim=(1, 2))
    union = (binary + text).sum(dim=(1, 2))
    has_text = text.sum(dim=(1, 2)) > 0
    dice = (1 - 2 * intersection / union.clamp_min(1e-6)) * has_text
    binary_loss = dice.sum() / has_text.sum().clamp_min(1)
    return probability_loss + threshold_weight * threshold_loss + binary_weight * binary_loss
