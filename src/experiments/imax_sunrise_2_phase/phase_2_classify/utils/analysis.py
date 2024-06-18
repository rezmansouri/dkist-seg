import torch
import numpy as np


def get_weights(masks):
    # masks = np.array(masks)
    # classes, counts = np.unique(masks, return_counts=True)
    # fqs = counts / sum(counts)
    # fqs = {k: v for k, v in zip(classes, fqs)}
    # w = torch.tensor([1/fqs[c] for c in sorted(classes)])
    # return w / max(w)
    return torch.tensor([1.0, 10.0, 10.0, 10.0, 1.0])
