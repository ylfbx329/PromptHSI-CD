import logging
import os
import random
from pathlib import Path

import clip
import numpy as np
import torch
from scipy.io import loadmat

from src.config.config import Config


def read_img(name):
    """
    读取数据并规范化
    数据类型：T1 T2 int16/float64，GT int16
    GT数值：-1 未知，0 未变化，1 变化
    :param name: 数据集名称
    :return: T1数据，T2数据，GT标签，预定义名称token
    """
    # T1数据，T2数据，GT标签
    root = Path(Config.args.proj_root, Config.args.data.root)
    if name == 'china farmland':
        # (420, 140, 154) (420, 140, 154) (420, 140)
        path = Path(root, 'Chinese farmland', 'China_Change_Dataset.mat')
        data = loadmat(str(path))
        t1, t2, gt = data['T1'].astype(np.float64), data['T2'].astype(np.float64), data['Binary'].astype(np.int16)
    elif name == 'usa':
        # (307, 241, 154) (307, 241, 154) (307, 241)
        # 1 Changed Pixels 2 Unchanged Pixels
        path = Path(root, 'American farmland')
        t1 = loadmat(str(Path(path, 'Sa1.mat')))['T1'].astype(np.float64)
        t2 = loadmat(str(Path(path, 'Sa2.mat')))['T2'].astype(np.float64)
        gt = loadmat(str(Path(path, 'SaGT.mat')))['GT'].astype(np.int16)
        gt[gt == 2] = 0
    elif name == 'river':
        # (463, 241, 198) (463, 241, 198) (463, 241)
        # 0 Unchanged Pixels 255 Changed Pixels
        path = Path(root, 'River')
        t1 = loadmat(str(Path(path, 'river_before.mat')))['river_before'].astype(np.float64)
        t2 = loadmat(str(Path(path, 'river_after.mat')))['river_after'].astype(np.float64)
        gt = loadmat(str(Path(path, 'groundtruth.mat')))['lakelabel_v1'].astype(np.int16)
        gt[gt == 255] = 1
    elif name == 'santa barbara':
        # (984, 740, 224) (984, 740, 224) (984, 740)
        # 0 Unknown Pixels 1 Changed Pixels 2 Unchanged Pixels
        path = Path(root, 'SantaBarbara BayArea and Hermiston', 'SantaBarbara', 'mat')
        t1 = loadmat(str(Path(path, 'barbara_2013.mat')))['HypeRvieW'].astype(np.float64)
        t2 = loadmat(str(Path(path, 'barbara_2014.mat')))['HypeRvieW'].astype(np.float64)
        gt = loadmat(str(Path(path, 'barbara_gtChanges.mat')))['HypeRvieW'].astype(np.int16)
        gt[gt == 0] = -1
        gt[gt == 2] = 0
    elif name == 'bay area':
        # (600, 500, 224) (600, 500, 224) (600, 500)
        # 0 Unknown Pixels 1 Changed Pixels 2 Unchanged Pixels
        path = Path(root, 'SantaBarbara BayArea and Hermiston', 'BayArea', 'mat')
        t1 = loadmat(str(Path(path, 'Bay_Area_2013.mat')))['HypeRvieW'].astype(np.float64)
        t2 = loadmat(str(Path(path, 'Bay_Area_2015.mat')))['HypeRvieW'].astype(np.float64)
        gt = loadmat(str(Path(path, 'bayArea_gtChanges2.mat.mat')))['HypeRvieW'].astype(np.int16)
        gt[gt == 0] = -1
        gt[gt == 2] = 0
    elif name == 'hermiston':
        # (390, 200, 242) (390, 200, 242) (390, 200)
        # usa farmland的另一版本，类似地物
        # 0 Unchanged Pixels 1 2 3 4 5 Changed Pixels
        path = Path(root, 'SantaBarbara BayArea and Hermiston', 'Hermiston')
        t1 = loadmat(str(Path(path, 'hermiston2004.mat')))['HypeRvieW'].astype(np.float64)
        t2 = loadmat(str(Path(path, 'hermiston2007.mat')))['HypeRvieW'].astype(np.float64)
        gt = loadmat(str(Path(path, 'rdChangesHermiston_5classes.mat')))['gt5clasesHermiston'].astype(np.int16)
        gt[gt > 1] = 1
    elif name == 'sigma china farmland':
        # (450, 140, 155) (450, 140, 155) (450, 140)
        # 1 Unchanged Pixels 2 Changed Pixels
        path = Path(root, 'HyperSIGMA', 'Chinese farmland')
        t1 = loadmat(str(Path(path, 'Farm1.mat')))['imgh'].astype(np.float64)
        t2 = loadmat(str(Path(path, 'Farm2.mat')))['imghl'].astype(np.float64)
        gt = loadmat(str(Path(path, 'GTChina1.mat')))['label'].astype(np.int16)
        gt[gt == 1] = 0
        gt[gt == 2] = 1
    else:
        raise ValueError(f'name must be in ["china farmland",] but got {name}')
    t1_neg_num = np.where(t1 < 0)[0].shape[0]
    t2_neg_num = np.where(t2 < 0)[0].shape[0]
    logging.info(f't1 neg num: {t1_neg_num}, t2 neg num: {t2_neg_num}')
    return t1, t2, gt


def read_text(text_filename):
    """

    :param text_filename:
    :return: 预定义名称token
    """
    root = Path(Config.args.proj_root, Config.args.data.root)
    # 标签名称token
    with open(os.path.join(root, text_filename), 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f]
    prompt_list = [f'remote sensing image patch for change detection: background.']
    for class_name in class_names:
        prompt_list.append(f'remote sensing image patch for change detection: {class_name}.')
    text_token = clip.tokenize(prompt_list)  # 57,77
    return text_token


def norm(img):
    """
    通道均值和标准差归一化
    :param img: (h.w.c)
    :return: 均值标准差归一化后的图像
    """
    eps = np.finfo(img.dtype).eps

    # 0-1 归一化
    # 计算每个通道的最大最小值
    min = np.min(img, axis=(0, 1))  # (c,) 向量，每个通道的均值
    max = np.max(img, axis=(0, 1))  # (c,) 向量，每个通道的标准差

    # 对每个通道进行标准化
    img_norm = (img - min) / np.maximum(max - min, eps)
    return img_norm


def stratified_split(label, train_ratio):
    """
    分层采样，采样后各类比例与采样前相同，仅能搭配重叠patch使用
    采样结果：采样（未）变化点=完整数据（未）变化点*train_ratio，采样变化点：采样未变化点=完整数据变化点：完整数据未变化点
    :param label: 标签数据
    :param train_ratio: 训练集比例，float类型
    :return: 训练集索引、验证集索引、测试集索引
    """
    label = np.ravel(label)
    total_index = np.where(label != -1)[0]
    change_index = np.where(label == 1)[0]
    not_change_index = np.where(label == 0)[0]
    np.random.shuffle(change_index)
    np.random.shuffle(not_change_index)
    # 分层采样
    train_change_num = int(train_ratio * len(change_index))
    train_not_change_num = int(train_ratio * len(not_change_index))
    # 用于训练
    train_index = np.concatenate([change_index[:train_change_num], not_change_index[:train_not_change_num]])
    # 用于评估并计算指标
    val_index = np.concatenate([change_index[train_change_num:], not_change_index[train_not_change_num:]])
    # 用于可视化
    test_index = total_index
    # 避免同类索引堆积，检验数据准确性时移除
    np.random.shuffle(train_index)
    return train_index, val_index, test_index


def text_collate_fn(data):
    t1, t2, label, text = zip(*data)
    t1, t2, label = torch.stack(t1), torch.stack(t2), torch.stack(label)
    text = text[0]
    return t1, t2, label, text


def worker_init_fn(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
