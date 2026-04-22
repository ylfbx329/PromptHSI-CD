import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import v2

from src.config.config import Config
from src.data.dataset import HyperCDDataset
from src.data.utils import norm, stratified_split, worker_init_fn, text_collate_fn, read_img, read_text


def get_cd_dataloader():
    """
    :return:
    train_loader：按配置文件随机抽取的小训练样本，用于训练
    val_loader：除去训练样本的测试样本，用于测试，得到指标
    test_loader：完整的所有数据样本，用于画图
    """
    logging.info('Start Data load...')
    data_param = Config.args.data

    # 原始高光谱
    # t1 t2 (h, w, c)
    t1, t2, gt = read_img(name=data_param.name)
    text = read_text(text_filename=data_param.text)

    # 标准化
    t1_scaler = norm(t1)
    t2_scaler = norm(t2)

    # padding
    # t1_pad t2_pad (h + margin * 2, w + margin * 2, pca_components)
    patch_size = data_param.patch_size
    margin = patch_size // 2
    t1_pad = np.pad(t1_scaler, pad_width=((margin,), (margin,), (0,)), mode='reflect')
    t2_pad = np.pad(t2_scaler, pad_width=((margin,), (margin,), (0,)), mode='reflect')

    # 修改图像为torch样式(c,h,w) (pca_components, h + margin * 2, w + margin * 2)
    t1_tensor = torch.from_numpy(t1_pad).to(torch.float32).permute(2, 0, 1)  # c h w
    t2_tensor = torch.from_numpy(t2_pad).to(torch.float32).permute(2, 0, 1)
    gt_tensor = torch.from_numpy(gt).to(torch.int64)

    # split，同时去除未知变化点（标签值为-1）
    train_ratio = data_param.train_ratio
    if data_param.split == 'stratify':
        train_index, val_index, test_index = stratified_split(gt, train_ratio)
    else:
        raise ValueError(f'split must be in ["stratify"] but got {data_param.split}')

    # get dataset and dataloader
    transform = v2.Compose([
        v2.RandomRotation(360),
        v2.RandomHorizontalFlip(),
        v2.RandomVerticalFlip()
    ])
    train_set = HyperCDDataset(t1_tensor, t2_tensor, gt_tensor, text, train_index, transform)
    val_set = HyperCDDataset(t1_tensor, t2_tensor, gt_tensor, text, val_index)
    test_set = HyperCDDataset(t1_tensor, t2_tensor, gt_tensor, text, test_index)

    train_batch_size = Config.args.train.batch_size
    val_batch_size = Config.args.val.batch_size
    test_batch_size = Config.args.test.batch_size
    # os.name == 'posix' 表示类Unix系统
    # num_workers的通用设置方法，Windows为0，Linux为CPU逻辑核心数和batch_size的小值，最大不超过8
    num_workers = min([os.cpu_count(), train_batch_size if train_batch_size > 1 else 0, 8]) if os.name == 'posix' else 0
    # 当使用GPU训练时pin_memory设置为True，提高数据加载效率
    pin_memory = True if 'cuda' in Config.args.device else False
    # 避免数据加载的随机性
    g = torch.Generator()
    g.manual_seed(Config.args.seed)
    train_loader = DataLoader(train_set,
                              batch_size=train_batch_size,
                              shuffle=True,
                              num_workers=num_workers,
                              collate_fn=text_collate_fn,
                              pin_memory=pin_memory,
                              worker_init_fn=worker_init_fn,
                              generator=g)
    val_loader = DataLoader(val_set,
                            batch_size=val_batch_size,
                            shuffle=False,
                            num_workers=num_workers,
                            collate_fn=text_collate_fn,
                            pin_memory=pin_memory,
                            worker_init_fn=worker_init_fn,
                            generator=g)
    test_loader = DataLoader(test_set,
                             batch_size=test_batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             collate_fn=text_collate_fn,
                             pin_memory=pin_memory,
                             worker_init_fn=worker_init_fn,
                             generator=g)

    logging.info(f'All sample {gt.size}, train sample {len(train_set)}, val sample {len(val_set)}, test sample {len(test_set)}')
    logging.info('Data load complete.')
    return train_loader, val_loader, test_loader
