import logging

import numpy as np
from matplotlib import pyplot as plt

from src.config.config import Config
from src.data.utils import read_img
from src.utils.utils import get_output_path


def plot(x, y, xlabel, ylabel, image_filename):
    """
    绘制简单的折线图
    :param x: 折线点的x轴坐标列表
    :param y: 折线点的y轴坐标列表
    :param xlabel: x轴标签
    :param ylabel: y轴标签
    :param image_filename: 用于保存的图像文件名
    """
    plt.figure(layout='constrained')
    plt.plot(x, y)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    title = image_filename.rsplit('.', 1)[0]
    plt.title(title)
    # 避免折线贴边，使图像美观
    plt.xlim([x[0] - 2, x[-1] + 2])
    plt.savefig(get_output_path(filename=image_filename, filetype='result'))
    plt.show()


def visual_change_detection(pred, pdf=False):
    _, _, gt = read_img(name=Config.args.data.name)
    all_result = np.full_like(gt, -1)
    mask = (gt != -1)
    all_result[mask] = pred

    # 初始化一个 RGB 图像
    height, width = all_result.shape
    pred_color = np.zeros((height, width, 3), dtype=np.uint8)
    gt_color = np.zeros((height, width, 3), dtype=np.uint8)

    # 定义颜色
    color_tp = [255, 255, 255]  # 白色
    color_tn = [0, 0, 0]  # 黑色
    color_fp = [0, 255, 0]  # 绿色
    color_fn = [255, 0, 0]  # 红色
    color_unknown = [190, 190, 190]  # 灰色

    # 遍历每种情况并赋值颜色
    gt_color[gt == -1] = color_unknown
    gt_color[gt == 0] = color_tn
    gt_color[gt == 1] = color_tp

    pred_color[(all_result == 1) & (gt == 1)] = color_tp  # TP
    pred_color[(all_result == 0) & (gt == 0)] = color_tn  # TN
    pred_color[(all_result == 1) & (gt == 0)] = color_fp  # FP
    pred_color[(all_result == 0) & (gt == 1)] = color_fn  # FN
    pred_color[gt == -1] = color_unknown  # 未知

    logging.info(f'color tp: {np.where((all_result == 1) & (gt == 1))[0].shape}')
    logging.info(f'color tn: {np.where((all_result == 0) & (gt == 0))[0].shape}')
    logging.info(f'color fp: {np.where((all_result == 1) & (gt == 0))[0].shape}')
    logging.info(f'color fn: {np.where((all_result == 0) & (gt == 1))[0].shape}')

    plt.figure(layout='constrained', dpi=600)
    plt.imshow(gt_color, cmap='gray', interpolation='none')
    plt.axis('off')
    suffix = 'pdf' if pdf else 'png'
    gt_img_path = get_output_path(filename=f'{Config.args.test.ckpt.split(".")[0]}-gt.{suffix}', filetype='result')
    plt.savefig(gt_img_path, dpi=600, bbox_inches="tight")
    plt.show()

    plt.figure(layout='constrained', dpi=600)
    plt.imshow(pred_color, interpolation='none')
    plt.axis('off')
    suffix = 'pdf' if pdf else 'png'
    pred_img_path = get_output_path(filename=f'{Config.args.test.ckpt.split(".")[0]}-pred.{suffix}', filetype='result')
    plt.savefig(pred_img_path, dpi=600, bbox_inches="tight")

    logging.info(f'Save pred image at {pred_img_path} and ground truth image at {gt_img_path}')
    plt.show()
