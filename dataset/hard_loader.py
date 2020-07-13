import json
from torch.utils.data import DataLoader
from .Dataloaders import cityscapes_processed_hard_extend


def get_val_loaders_extend(data_root, p_num=600, cp_num=60):
    opts = json.load(open('dataset/Experiments/gnn-active-spline.json', 'r'))
    opts["dataset"]["train_val"]["data_dir"] = data_root
    opts["dataset"]["train_val"]["p_num"] = p_num
    opts["dataset"]["train_val"]["cp_num"] = cp_num

    dataset_val = cityscapes_processed_hard_extend.DataProvider(split='val',
                                                                opts=opts['dataset']['train_val'],
                                                                mode="test")

    train_val_bs = opts["dataset"]["train_val"]["batch_size"]

    val_loader = DataLoader(dataset_val, batch_size=train_val_bs,
                            shuffle=False, num_workers=opts["dataset"]["train_val"]["num_workers"],
                            collate_fn=cityscapes_processed_hard_extend.collate_fn, drop_last=True)

    return val_loader


def get_data_loaders_extend(data_root, p_num=600, cp_num=60):
    opts = json.load(open('dataset/Experiments/gnn-active-spline.json', 'r'))
    opts["dataset"]["train"]["data_dir"] = data_root
    opts["dataset"]["train_val"]["data_dir"] = data_root
    opts["dataset"]["train"]["p_num"] = p_num
    opts["dataset"]["train"]["cp_num"] = cp_num
    opts["dataset"]["train_val"]["p_num"] = p_num
    opts["dataset"]["train_val"]["cp_num"] = cp_num

    dataset_train = cityscapes_processed_hard_extend.DataProvider(split='train',
                                                                  opts=opts['dataset']['train'])
    dataset_val = cityscapes_processed_hard_extend.DataProvider(split='train_val', opts=opts['dataset']['train_val'])

    train_bs = opts["dataset"]["train"]["batch_size"]
    train_val_bs = opts["dataset"]["train_val"]["batch_size"]

    train_loader = DataLoader(dataset_train, batch_size=train_bs,
                              shuffle=True, num_workers=opts["dataset"]["train"]["num_workers"],
                              collate_fn=cityscapes_processed_hard_extend.collate_fn, drop_last=True)

    val_loader = DataLoader(dataset_val, batch_size=train_val_bs,
                            shuffle=False, num_workers=opts["dataset"]["train_val"]["num_workers"],
                            collate_fn=cityscapes_processed_hard_extend.collate_fn, drop_last=True)

    return train_loader, val_loader

