import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score


def multilabel_metrics(y_true, y_prob, labels, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )
    result = {}
    for i, label in enumerate(labels):
        try:
            roc = roc_auc_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            roc = float("nan")
        try:
            pr = average_precision_score(y_true[:, i], y_prob[:, i])
        except ValueError:
            pr = float("nan")
        result[label] = {
            "precision": float(p[i]), "recall": float(r[i]), "f1": float(f1[i]),
            "roc_auc": float(roc), "pr_auc": float(pr)
        }
    result["macro_f1"] = float(np.mean(f1))
    return result
