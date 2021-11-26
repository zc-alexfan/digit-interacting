import numpy as np
import torch


def segm_iou(pred, target, n_classes, tol, background_cls=0):
    """
    Compute segmentation iou for a segmentation mask.
    pred: (dim, dim)
    target: (dim, dim)
    n_classes: including the background class
    tol: how many entries to ignore for union for noisy target map
    """
    assert isinstance(pred, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert pred.shape == target.shape
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
    for cls in range(n_classes):
        if cls == background_cls:
            continue
        pred_cls = pred == cls
        target_cls = target == cls
        target_inds = target_cls.nonzero(as_tuple=True)[0]
        intersection = pred_cls[target_inds].long().sum()
        union = pred_cls.long().sum() + target_cls.long().sum() - intersection

        if union < tol:
            # If there is no ground truth, do not include in evaluation
            ious.append(float('nan'))
        else:
            ious.append(float(intersection) / float(max(union, 1)))
    return float(np.nanmean(np.array(ious)))
