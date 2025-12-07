from sklearn.metrics import accuracy_score,roc_auc_score,f1_score,precision_score,confusion_matrix,roc_curve,recall_score
import numpy as np
import json
def save_json(fname,obj):
         with open(fname, "w") as f:
                json.dump(obj,f)

def alpha_tuner(model_name, model, X_val, y_val):

    threshold_results = []
    alpha_arr = np.linspace(0, 1, 100)

    # Compute probabilities once
    yhat_prob = model.predict_proba(X_val)[:, 1]

    for alpha in alpha_arr:
        yhat = (yhat_prob >= alpha).astype(int)

        threshold_results.append({
            "alpha": float(alpha),
            "recall": float(recall_score(y_val, yhat)),
            "accuracy": float(accuracy_score(y_val, yhat)),
            "auc": float(roc_auc_score(y_val, yhat_prob)),
            "precision": float(precision_score(y_val, yhat, zero_division=0)),
            "f1": float(f1_score(y_val, yhat)),
        })

    # Save whole sweep once
    save_json(
        f"tuning_results/thresholds/{model_name}_alpha_sweep.json",
        threshold_results
    )

    # ----- MAXIMIZE F2 SCORE (RECALL-WEIGHTED) -----

    beta = 2  # F2 score = recall-weighted
    f2_scores = []

    for res in threshold_results:
        p = res["precision"]
        r = res["recall"]

        if p + r == 0:
            f2 = 0.0
        else:
            f2 = (1 + beta**2) * (p * r) / (beta**2 * p + r)

        f2_scores.append(f2)

    best_idx = int(np.argmax(f2_scores))

    best_alpha = threshold_results[best_idx]["alpha"]
    best_f2 = f2_scores[best_idx]
    best_recall = threshold_results[best_idx]["recall"]
    best_precision = threshold_results[best_idx]["precision"]

    save_json(
        f"tuning_results/thresholds/{model_name}_best_alpha.json",
        {
            "best_alpha": best_alpha,
            "best_f2": best_f2,
            "best_recall": best_recall,
            "best_precision": best_precision
        }
    )

    return best_alpha

