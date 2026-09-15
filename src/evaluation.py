## Metrik hesaplama, threshold optimizasyonu ve sonuç raporlama.

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

def evaluate_predictions(y_true, y_pred) -> dict:
    """Model performans metriklerini hesaplar."""
    acc = accuracy_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, output_dict=True)
    return {"accuracy": acc, "confusion_matrix": cm, "report": report}

def optimize_threshold(y_true, proba_pos, min_thresh=0.30, max_thresh=0.46, step=0.01, min_recall=0.85):
    """Belirli bir minimum recall hedefi altında en uygun F1 eşiğini arar."""
    best_thresh = 0.5
    best_f1, best_recall, best_precision = 0, 0, 0
    records = []

    for thresh in np.arange(min_thresh, max_thresh, step):
        y_pred = (proba_pos >= thresh).astype(int)
        rep = classification_report(y_true, y_pred, output_dict=True)["1"]
        rec, prec, f1 = rep["recall"], rep["precision"], rep["f1-score"]
        records.append({"threshold": thresh, "precision": prec, "recall": rec, "f1": f1})

        if rec >= min_recall and (f1 > best_f1 or (rec > best_recall and prec > best_precision)):
            best_thresh = thresh
            best_f1, best_recall, best_precision = f1, rec, prec

    return best_thresh, pd.DataFrame(records)