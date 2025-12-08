import numpy as np 
from sklearn.metrics import precision_recall_curve


def tune_alpha_f1(y_true, y_proba, pos_label=1):

    precision, recall, thresholds = precision_recall_curve(
        y_true, y_proba, pos_label=pos_label
    )

    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    # So we compute F1 only for those indices.
    f1_scores = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]

    best_metrics = {
        "threshold": float(best_threshold),
        "f1": float(f1_scores[best_idx]),
        "precision": float(precision[best_idx]),
        "recall": float(recall[best_idx]),
    }

    return best_threshold, 


