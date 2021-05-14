##################################################
# Imports
##################################################

import torch
import torch.nn.functional as F

def accuracy(preds, labels, preds_with_logits=True):
    """
    Compute the accuracy.
    Args:
        preds: tensor of shape [bs, num_classes(, ...)].
        labels: tensor of shape [bs(, ...)].
        preds_with_logits: bool.
    Output:
        acc: scalar.
    """
    if preds_with_logits:
        preds = F.softmax(preds, 1)
    preds = preds.argmax(1)
    acc = (1.0 * (preds == labels)).mean()
    return acc
