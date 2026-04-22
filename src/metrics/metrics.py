import logging

import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, jaccard_score, cohen_kappa_score, roc_curve, auc


def metrics(y_true, y_pred, y_pred_prob=None, print_info=True):
    """
    计算指标、打印为日志并返回指标结果
    :param y_true: 真值标签
    :param y_pred: 模型预测结果
    :param y_pred_prob: 模型原始输出，(n, 2) n=数据集大小
    :param print_info:
    """
    y_true = y_true.ravel()
    y_pred = y_pred.ravel()
    data_idx = np.where(y_true != -1)[0]
    y_true = y_true[data_idx]
    y_pred = y_pred[data_idx]
    oa = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    iou = jaccard_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred)

    # 打印指标
    if print_info:
        logging.info(f"Overall Accuracy (OA):{oa}")
        logging.info(f"Precision:{precision}")
        logging.info(f"Recall:{recall}")
        logging.info(f"F1-score:{f1}")
        logging.info(f"IoU(Jaccard):{iou}")
        logging.info(f"Kappa:{kappa}")

    if y_pred_prob is not None:
        max_prob, idx = torch.sigmoid(torch.Tensor(y_pred_prob)).max(dim=1)
        fpr, tpr, _ = roc_curve(y_true, max_prob)
        auc_value = auc(fpr, tpr)
        logging.info(f'AUC:{auc_value}')
    else:
        auc_value = -1

    return oa, precision, recall, f1, iou, kappa, auc_value
