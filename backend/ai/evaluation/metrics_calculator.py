# Tính F1, Precision, Recall, AUC cho từng task
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
import numpy as np

def calc_metrics(y_true, y_pred, task_type='multiclass', average='macro'):
    metrics = {}
    if task_type == 'multiclass':
        metrics['f1'] = f1_score(y_true, y_pred, average=average)
        metrics['precision'] = precision_score(y_true, y_pred, average=average)
        metrics['recall'] = recall_score(y_true, y_pred, average=average)
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred, multi_class='ovr', average=average)
        except Exception:
            metrics['auc'] = None
    elif task_type == 'multilabel':
        metrics['f1'] = f1_score(y_true, y_pred, average=average)
        metrics['precision'] = precision_score(y_true, y_pred, average=average)
        metrics['recall'] = recall_score(y_true, y_pred, average=average)
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred, average=average)
        except Exception:
            metrics['auc'] = None
    return metrics
