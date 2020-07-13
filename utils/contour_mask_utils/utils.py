import numpy as np
import cv2
import torch


def apply_mask(img, mask, color=(1.0, 0.0, 0.0), alpha=0.3):
    image = img

    for n, c in enumerate(color):
        image[:, :, n] = np.where(
            mask > 0.5,
            image[:, :, n] * (1.0 - alpha) + alpha * c,
            image[:, :, n]
        )
    return image


def erode_dilate(mask):
    out = mask.astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    dilation = cv2.dilate(out, kernel)
    erosion = cv2.erode(dilation, kernel)
    out = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)  # remove small region
    return out


def iou_from_poly(pred, gt, width, height):
    masks = np.zeros((2, height, width), dtype=np.uint8)

    if not isinstance(pred, list):
        pred = [pred]
    if not isinstance(gt, list):
        gt = [gt]

    for p in pred:
        masks[0] = draw_poly(masks[0], p)

    for g in gt:
        masks[1] = draw_poly(masks[1], g)

    return iou_from_mask(masks[0], masks[1]), masks


def iou_from_mask(pred, gt):
    """
    Compute intersection over the union.
    Args:
        pred: Predicted mask
        gt: Ground truth mask
    """
    pred = pred.astype(np.bool)
    gt = gt.astype(np.bool)

    # true_negatives = np.count_nonzero(np.logical_and(np.logical_not(gt), np.logical_not(pred)))
    false_negatives = np.count_nonzero(np.logical_and(gt, np.logical_not(pred)))
    false_positives = np.count_nonzero(np.logical_and(np.logical_not(gt), pred))
    true_positives = np.count_nonzero(np.logical_and(gt, pred))

    union = float(true_positives + false_positives + false_negatives)
    intersection = float(true_positives)

    iou = intersection / union if union > 0. else 0.

    return iou


def draw_poly(mask, poly):
    """
    NOTE: Numpy function
    Draw a polygon on the mask.
    Args:
    mask: np array of type np.uint8
    poly: np array of shape N x 2
    """
    if not isinstance(poly, np.ndarray):
        poly = np.array(poly)
    cv2.fillPoly(mask, [poly], 1)
    return mask


def gather_feature(id, feature):
    feature_id = id.unsqueeze_(2).long().expand(id.size(0), id.size(1), feature.size(2))
    cnn_out = torch.gather(feature, 1, feature_id).float()
    return cnn_out


def poly01_to_poly0g(poly, grid_size):
    """
    [0, 1] coordinates to [0, grid_size] coordinates
    Note: simplification is done at a reduced scale
    """
    poly = np.floor(poly * grid_size).astype(np.int32)
    poly = cv2.approxPolyDP(poly, 0, False)[:, 0, :]

    return poly


def create_adjacency_matrix_cat(batch_size, n_adj, n_nodes):
    a = np.zeros([batch_size, n_nodes, n_nodes])

    for t in range(batch_size):
        for i in range(n_nodes):
            for j in range(-n_adj // 2, n_adj // 2 + 1):
                if j != 0:
                    a[t][i][(i + j) % n_nodes] = 1
                    a[t][(i + j) % n_nodes][i] = 1

    return a.astype(np.float32)
