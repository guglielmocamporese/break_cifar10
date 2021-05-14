##################################################
# Imports
##################################################

import torch
import torch.nn.functional as F


def cross_entropy(logits, labels, smooth=0.0):
    """
    Compute the cross entropy.
    Args:
        logits: tensor of shape [bs, num_classes(, ...)].
        labels: tensor of shape [bs(, ...)].
        smooth: scalar, for label smoothing.
    Output:
        xent: scalar.
    """

    assert (0.0 <= smooth) and (smooth <= 1.0)
    if smooth > 0:
        num_classes = logits.shape[1]
        labels_oh = F.one_hot(labels, num_classes)
        labels_oh = labels_oh.unsqueeze(1).transpose(1, -1).squeeze(-1)
        labels_smooth = (1.0 - smooth) * labels_oh + smooth / num_classes
        xent = (- labels_smooth * F.log_softmax(logits, 1)).sum(1).mean()

    else:
        xent = F.cross_entropy(logits, labels)
    return xent
