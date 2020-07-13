from skimage.io import imread
import skimage.color as color
import cv2
import os
import numpy as np
import random
import torch
from scipy.ndimage import distance_transform_edt as distance


def create_folder(path):
    if os.path.exists(path):
        pass
    else:
        os.system('mkdir -p %s' % (path))
        print('Experiment folder created at: %s' % (path))


def rgb_img_read(img_path):
    """
    Read image and always return it as a RGB image (3D vector with 3 channels).
    """
    img = imread(img_path)
    if len(img.shape) == 2:
        img = color.gray2rgb(img)

    # Deal with RGBA
    img = img[..., :3]

    if img.dtype == 'uint8':
        # [0,1] image
        img = img.astype(np.float32) / 255
    return img


def get_full_mask_from_instance(min_area, instance):
    img_h, img_w = instance['img_height'], instance['img_width']
    mask = np.zeros((img_h, img_w), dtype=np.uint8)
    for component in instance['components']:
        p = np.array(component['poly'], np.int)
        if component['area'] < min_area:
            continue
        else:
            draw_poly(mask, p)
    return mask


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
    cv2.fillPoly(mask, [poly], 255)
    return mask


def poly01_to_poly0g(poly, grid_size):
    """
    [0, 1] coordinates to [0, grid_size] coordinates
    Note: simplification is done at a reduced scale
    """
    poly = np.floor(poly * grid_size).astype(np.int32)
    poly = cv2.approxPolyDP(poly, 0, False)[:, 0, :]

    return poly


def get_vertices_mask(poly, mask):
    """
    Generate a vertex mask
    """
    mask[poly[:, 1], poly[:, 0]] = 1.

    return mask


def get_edge_mask(poly, mask):
    """
    Generate edge mask
    """
    cv2.polylines(mask, [poly], True, [1])

    return mask


# #used for Surface loss assert
def checkSimplex(probs, axis=1):
    _sum = probs.sum(axis).type(torch.float32)  # ok for python3.6
    # _sum = torch.zeros(probs.shape[0], probs.shape[2], probs.shape[3])
    # # # same as: _sum = torch.einsum('bcwh -> bwh', [probs])
    # for b in range(probs.shape[0]):
    #     for w in range(probs.shape[2]):
    #         for h in range(probs.shape[3]):
    #             _sum[b][w][h] = torch.sum(probs[b, :, w, h])

    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)  # works fine for torch 0.4.1v , check if |param1-param2|<=atol+rtol*|param2|


def checkSimplex_(probs, axis=1):
    # _sum = probs.sum(axis).type(torch.float32) #ok for python3.6
    print(probs.shape)
    _sum = torch.zeros(probs.shape[0], probs.shape[2], probs.shape[3]).type(torch.float32).cuda()
    # # same as: _sum = torch.einsum('bcwh -> bwh', [probs])
    for b in range(probs.shape[0]):
        for w in range(probs.shape[2]):
            for h in range(probs.shape[3]):
                _sum[b][w][h] = torch.sum(probs[b, :, w, h])

    _ones = torch.ones_like(_sum, dtype=torch.float32).cuda()
    return torch.allclose(_sum, _ones)  # works fine for torch 0.4.1v , check if |param1-param2|<=atol+rtol*|param2|


def uniq(pa):  # pa: Tensor
    return set(torch.unique(pa.cpu()).numpy())  # return the unique set, eg.[3, 2, 2, 4, 3] to [3, 2, 4]


def sset(pa, pb):  # pa:Tensor, pb:Iterable
    return uniq(pa).issubset(pb)


def one_hot(pa, axis=1):
    return checkSimplex(pa, axis) and sset(pa, [0, 1])


# #encode the mask to one-hot value
def class2one_hot(seg, C):  # seg:Tensor; C:int
    if len(seg.shape) == 2:
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    b, w, h, = seg.shape
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.float32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


# #calculate the distance matric by one-hot encode of mask
def one_hot2dist(seg):  # seg:np.ndarray
    assert one_hot(torch.Tensor(seg), axis=0)
    C = len(seg)  # int
    # res = torch.zeros_like(seg)
    seg = seg.numpy()
    res = np.zeros_like(seg)
    # convert res to numpy.ndarry
    for c in range(C):
        posmask = seg[c].astype(np.bool)
        if posmask.any():
            negmask = ~posmask
            res[c] = distance(negmask) * negmask - (distance(posmask) - 1) * posmask
    res = torch.from_numpy(res)
    return res


def extreme_points(polygon, pert=0):

    def find_point(ids):
        sel_id = ids[0][random.randint(0, len(ids[0]) - 1)]
        return sel_id

    x = polygon[:,0]
    y = polygon[:,1]

    ex_0 = find_point(np.where(y  >= np.max(y) - pert))
    ex_1 = find_point(np.where(y  <= np.min(y) + pert))
    ex_2 = find_point(np.where(x  >= np.max(x) - pert))
    ex_3 = find_point(np.where(x  <= np.min(x) + pert))
    return polygon[ex_0], polygon[ex_1],  polygon[ex_2], polygon[ex_3]


def make_gt(labels, sigma=10, h=224, w=224):
    """ Make the ground-truth for  landmark.
    img: the original color image
    labels: label with the Gaussian center(s) [[x0, y0],[x1, y1],...]
    sigma: sigma of the Gaussian.
    one_mask_per_point: masks for each point in different channels?
    """
    if labels is None:
        gt = make_gaussian((h, w), center=(h//2, w//2), sigma=sigma)
    else:
        labels = np.array(labels)

        if labels.ndim == 1:
            labels = labels[np.newaxis]
        gt = np.zeros(shape=(h, w), dtype=np.float64)
        for ii in range(labels.shape[0]):
            gt = np.maximum(gt, make_gaussian((h, w), center=labels[ii, :], sigma=sigma))

    gt = gt.astype(dtype=np.float32)

    return gt


def make_gaussian(size, sigma=10, center=None, d_type=np.float64):
    """ Make a square gaussian kernel.
    size: is the dimensions of the output gaussian
    sigma: is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)
    y = y[:, np.newaxis]

    if center is None:
        x0 = y0 = size[0] // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / sigma ** 2).astype(d_type)
