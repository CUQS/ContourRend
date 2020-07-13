import glob
import json
import multiprocessing.dummy as multiprocessing
import os.path as osp
import random

import numpy as np
import skimage.transform as transform
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageDraw

import dataset.Utils.utils as utils

EPS = 1e-7


def process_info(args):
    """
    Process a single json file
    """
    fname, opts = args

    with open(fname, 'r') as f:
        ann = json.load(f)

    examples = []
    skipped_instances = 0

    for instance in ann:
        components = instance['components']

        if opts['class_filter'] is not None and instance['label'] not in opts['class_filter']:
            continue

        # candidates = [c for c in components if len(c['poly']) >= opts['min_poly_len']]
        candidates = [c for c in components]

        if opts['sub_th'] is not None:
            total_area = np.sum([c['area'] for c in candidates])
            candidates = [c for c in candidates if c['area'] > opts['sub_th'] * total_area]

        candidates = [c for c in candidates if c['area'] >= opts['min_area']]

        if opts['skip_multicomponent'] and len(candidates) > 1:
            skipped_instances += 1
            continue

        instance['components'] = candidates
        if candidates:
            examples.append(instance)

    return examples, skipped_instances


def collate_fn(batch_list):
    keys = batch_list[0].keys()
    collated = {}

    for key in keys:
        val = [item[key] for item in batch_list]

        t = type(batch_list[0][key])

        if t is np.ndarray:

            if key != "orig_poly":
                try:
                    val = np.stack(val, axis=0)
                    val = torch.from_numpy(val)
                except:
                    # for items that are not the same shape
                    # for eg: orig_poly
                    val = [item[key] for item in batch_list]
            else:
                val = [item[key] for item in batch_list]

        collated[key] = val

    return collated


class DataProvider(Dataset):
    """
    Class for the data provider
    """

    def __init__(self, opts, split='train', mode='train_ce', debug=False):
        """
        split: 'train', 'train_val' or 'val'
        opts: options from the json file for the dataset
        """
        self.opts = opts
        self.mode = mode
        self.debug = debug
        print(self.opts.keys())
        print('Dataset Options: ', opts)

        if self.mode != 'tool':
            # in tool mode, we just use these functions
            self.data_dir = osp.join(opts['data_dir'], split)
            self.instances = []
            self.read_dataset()
            print('Read %d instances in %s split' % (len(self.instances), split))

    def read_dataset(self):
        data_list = glob.glob(osp.join(self.data_dir, '*/*.json'))
        data_list = [[d, self.opts] for d in data_list]
        if self.debug:
            data_list = data_list[:20]
        pool = multiprocessing.Pool(self.opts['num_workers'])
        data = pool.map(process_info, data_list)
        pool.close()
        pool.join()

        print("Dropped %d multi-component instances" % (np.sum([s for _, s in data])))

        self.instances = [instance for image, _ in data for instance in image]

        if self.debug:
            self.instances = self.instances[:2]

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.prepare_instance(idx)

    def prepare_instance(self, idx):
        """
        Prepare a single instance, can be both multicomponent
        or just a single component
        """
        instance = self.instances[idx]

        if self.opts['skip_multicomponent']:
            # Skip_multicomponent is true even during test because we use only
            # 1 bbox and no polys
            assert len(instance['components']) == 1, 'Found multicomponent instance\
            with skip_multicomponent set to True!'

            component = instance['components'][0]
            results = self.prepare_component(instance, component)

            if 'test' in self.mode:
                results['instance'] = instance

        else:
            if 'test' in self.mode:
                # if len(instance['components']) > 1:
                #     area_max = 0
                #     select_i = 0
                #     for comp_idx, comp_i in enumerate(instance['components']):
                #         if comp_i["area"] > area_max:
                #             area_max = comp_i["area"]
                #             select_i = comp_idx
                #     component = instance['components'][select_i]
                # else:
                #     component = instance['components'][0]
                component = instance['components'][0]
                results = self.prepare_component(instance, component)

                if self.opts['ext_points']:

                    all_comp_gt_poly = []
                    for component in instance['components']:
                        if component['area'] < self.opts['min_area']:
                            continue
                        else:
                            comp = self.extract_crop(component, instance, results['context_expansion'])
                            all_comp_gt_poly.extend(comp['poly'].tolist())

                    all_comp_gt_poly = np.array(all_comp_gt_poly) * self.opts['img_side']
                    ex_0, ex_1, ex_2, ex_3 = utils.extreme_points(all_comp_gt_poly)
                    nodes = [ex_0, ex_1, ex_2, ex_3]
                    point_annotation = utils.make_gt(nodes, h=self.opts['img_side'], w=self.opts['img_side'])
                    results['annotation_prior'] = point_annotation

            elif 'train' in self.mode:
                component = random.choice(instance['components'])
                results = self.prepare_component(instance, component)

            results['instance'] = instance

        return results

    def prepare_component(self, instance, component):
        """
        Prepare a single component within an instance
        """
        pnum = self.opts['p_num']
        cp_num = self.opts['cp_num']

        # create circle polygon data
        pointsnp = np.zeros(shape=(cp_num, 2), dtype=np.float32)
        for i in range(cp_num):
            thera = 1.0 * i / cp_num * 2 * np.pi
            x = np.cos(thera)
            y = -np.sin(thera)
            pointsnp[i, 0] = x
            pointsnp[i, 1] = y

        fwd_poly = (0.7 * pointsnp + 1) / 2

        arr_fwd_poly = np.ones((cp_num, 2), np.float32) * 0.
        arr_fwd_poly[:, :] = fwd_poly

        lo, hi = self.opts['random_context']
        context_expansion = random.uniform(lo, hi)

        crop_info = self.extract_crop(component, instance, context_expansion)

        img = crop_info['img']

        # get the onehot labels and dist_maps for boundary loss

        poly = crop_info['poly']

        orig_poly = poly.copy()

        gt_orig_poly = poly.copy()
        gt_orig_poly = utils.poly01_to_poly0g(gt_orig_poly, 28)
        # Get masks
        vertex_mask = np.zeros((28, 28), np.float32)
        edge_mask = np.zeros((28, 28), np.float32)
        vertex_mask = utils.get_vertices_mask(gt_orig_poly, vertex_mask)
        edge_mask = utils.get_edge_mask(gt_orig_poly, edge_mask)

        mask = np.asarray(crop_info['mask'])  # wh
        mask_tensor = torch.from_numpy(mask)
        onehot_label = utils.class2one_hot(mask_tensor, 2)[0]
        mask_distmap = utils.one_hot2dist(onehot_label)

        gt_poly = self.uniformsample(poly, pnum)
        arr_gt_poly = np.ones((pnum, 2), np.float32) * 0.
        arr_gt_poly[:, :] = gt_poly

        # Numpy doesn't throw an error if the last index is greater than size
        train_dict = {'orig_poly': orig_poly,
                      'gt_poly': arr_gt_poly,
                      'mask_distmap': np.array(mask_distmap)}

        boundry_dic = {
            'vertex_mask':vertex_mask,
            'edge_mask':edge_mask
        }
        train_dict.update(boundry_dic)
        train_dict['label'] = instance['label']

        # for Torch, use CHW, instead of HWC
        img = img.transpose(2, 0, 1)
        # blank_image
        return_dict = {
            'img': img,
            'gt_mask': crop_info['gt_mask'],
            'fwd_poly': arr_fwd_poly,
            'img_path': instance['img_path'],
            'patch_w': crop_info['patch_w'],
            'starting_point': crop_info['starting_point'],
            'context_expansion': context_expansion
        }

        return_dict.update(train_dict)

        return return_dict

    def uniformsample(self, pgtnp_px2, newpnum):

        pnum, cnum = pgtnp_px2.shape
        assert cnum == 2

        idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
        pgtnext_px2 = pgtnp_px2[idxnext_p]
        edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
        edgeidxsort_p = np.argsort(edgelen_p)

        # two cases
        # we need to remove gt points
        # we simply remove shortest paths
        if pnum > newpnum:
            edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
            edgeidxsort_k = np.sort(edgeidxkeep_k)
            pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
            assert pgtnp_kx2.shape[0] == newpnum
            return pgtnp_kx2
        # we need to add gt points
        # we simply add it uniformly
        else:
            edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
            for i in range(pnum):
                if edgenum[i] == 0:
                    edgenum[i] = 1

            # after round, it may has 1 or 2 mismatch
            edgenumsum = np.sum(edgenum)
            if edgenumsum != newpnum:

                if edgenumsum > newpnum:

                    id = -1
                    passnum = edgenumsum - newpnum
                    while passnum > 0:
                        edgeid = edgeidxsort_p[id]
                        if edgenum[edgeid] > passnum:
                            edgenum[edgeid] -= passnum
                            passnum -= passnum
                        else:
                            passnum -= edgenum[edgeid] - 1
                            edgenum[edgeid] -= edgenum[edgeid] - 1
                            id -= 1
                else:
                    id = -1
                    edgeid = edgeidxsort_p[id]
                    edgenum[edgeid] += newpnum - edgenumsum

            assert np.sum(edgenum) == newpnum

            psample = []
            for i in range(pnum):
                pb_1x2 = pgtnp_px2[i:i + 1]
                pe_1x2 = pgtnext_px2[i:i + 1]

                wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i];

                pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
                psample.append(pmids)

            psamplenp = np.concatenate(psample, axis=0)
            return psamplenp

    def extract_crop(self, component, instance, context_expansion):
        img = utils.rgb_img_read(instance['img_path'])
        mask = utils.get_full_mask_from_instance(100, instance)

        poly = np.array(component['poly'])

        xs = poly[:, 0]
        ys = poly[:, 1]

        bbox = instance['bbox']
        x0, y0, w, h = bbox

        x_center = x0 + (1 + w) / 2.
        y_center = y0 + (1 + h) / 2.

        widescreen = True if w > h else False

        if not widescreen:
            img = img.transpose((1, 0, 2))
            mask = mask.transpose((1, 0))
            x_center, y_center, w, h = y_center, x_center, h, w
            xs, ys = ys, xs

        x_min = int(np.floor(x_center - w * (1 + context_expansion) / 2.))
        x_max = int(np.ceil(x_center + w * (1 + context_expansion) / 2.))

        x_min = max(0, x_min)
        x_max = min(img.shape[1] - 1, x_max)

        patch_w = x_max - x_min
        # NOTE: Different from before

        y_min = int(np.floor(y_center - patch_w / 2.))
        y_max = y_min + patch_w

        top_margin = max(0, y_min) - y_min

        y_min = max(0, y_min)
        y_max = min(img.shape[0] - 1, y_max)

        scale_factor = float(self.opts['img_side']) / patch_w

        patch_img = img[y_min:y_max, x_min:x_max, :]
        patch_mask = mask[y_min:y_max, x_min:x_max]

        new_img = np.zeros([patch_w, patch_w, 3], dtype=np.float32)
        new_img[top_margin: top_margin + patch_img.shape[0], :, ] = patch_img

        new_img = transform.rescale(new_img, scale_factor, order=1, preserve_range=True, multichannel=True)
        new_img = new_img.astype(np.float32)
        # assert new_img.shape == [self.opts['img_side'], self.opts['img_side'], 3]

        new_mask = np.zeros([patch_w, patch_w], dtype=np.float32)
        new_mask[top_margin: top_margin+patch_img.shape[0], :] = patch_mask
        new_mask = transform.rescale(new_mask, scale_factor, order=1,
                                     preserve_range=True, multichannel=False,
                                     anti_aliasing=False)
        new_mask[new_mask > 0.99] = 1.0
        new_mask[new_mask < 1] = 0.0
        new_mask = new_mask.astype(np.float32)

        starting_point = [x_min, y_min - top_margin]

        xs = (xs - x_min) / float(patch_w)
        ys = (ys - (y_min - top_margin)) / float(patch_w)

        xs = np.clip(xs, 0 + EPS, 1 - EPS)
        ys = np.clip(ys, 0 + EPS, 1 - EPS)

        if not widescreen:
            # Now that everything is in a square
            # bring things back to original mode
            new_img = new_img.transpose((1, 0, 2))
            new_mask = new_mask.transpose((1, 0))
            starting_point = [y_min - top_margin, x_min]
            xs, ys = ys, xs

        return_dict = {
            'img': new_img,
            'patch_w': patch_w,
            'top_margin': top_margin,
            'patch_shape': patch_img.shape,
            'scale_factor': scale_factor,
            'starting_point': starting_point,
            'widescreen': widescreen
        }

        enlarged_poly = np.array([xs * scale_factor * patch_w, ys * scale_factor * patch_w]).T

        poly = np.array([xs, ys]).T
        return_dict['poly'] = poly

        poly_ = [tuple((item)) for item in enlarged_poly]  # note: item in origin_poly must be tuple type for 'imgdraw.polygon()' function
        enlarged_p = [list(item) for item in poly_]
        return_dict['enlarged_poly'] = enlarged_p
        mask = np.zeros((224, 224), dtype=np.uint8)
        mask = Image.fromarray(mask)
        imgdraw_mask_ = ImageDraw.Draw(mask)
        imgdraw_mask_.polygon(poly_, fill=1)
        return_dict['mask'] = mask
        return_dict['gt_mask'] = new_mask

        return return_dict
