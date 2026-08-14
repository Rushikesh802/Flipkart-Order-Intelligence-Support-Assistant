"""Baseline:DummyClassifier (most-frequent) on stratified 80/20 split,
and a Logistic Regression with class_weight='balanced' for comparison."""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from preprocess import get_fitted_data


def main():
    X_train, X_test, y_train, y_test, prep = get_fitted_data()

    # --- Baseline: DummyClassifier (most-frequent) ---
    dummy = DummyClassifier(strategy="most_frequent", random_state=42)
    dummy.fit(X_train, y_train)
    dummy_pred = dummy.predict(X_test)

    print("=== Baseline (Dummy: most_frequent) ===")
    print(f"  Accuracy: {accuracy_score(y_test, dummy_pred):.4f}")
    print(f"  F1 (returned=1): {f1_score(y_test, dummy_pred):.4f}")

    # --- Logistic Regression with class_weight="balanced" ---
    logit = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    logit.fit(X_train, y_train)

    # Default 0.5 threshold
    y_pred = logit.predict(X_test)
    y_proba = logit.predict_proba(X_test)[:, 1]

    print("\n=== Logistic Regression (balanced, threshold=0.50) ===")
    print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"  F1:        {f1_score(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_proba):.4f}")

    # --- Threshold sweep: 0.10 to 0.90 inclusive, step 0.02 ---
    thresholds = [round(0.10 + 0.02 * i, 2) for i in range(41)]  # 0.10..0.90

    # Tabulate full sweep
    print("\n=== Threshold sweep: F1 / precision / recall vs threshold ===")
    print("threshold\tF1\tprecision\trecall")
    best_f1 = -1.0
    best_threshold = 0.5
    best_precision = 0.0
    best_recall = 0.0
    rows = []
    for threshold in thresholds:
        y_pred_t = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_test, y_pred_t)
        precision = precision_score(y_test, y_pred_t, zero_division=0)
        recall = recall_score(y_test, y_pred_t)
        rows.append((threshold, f1, precision, recall))
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
            best_precision = precision
            best_recall = recall

    for threshold, f1, precision, recall in rows:
        print(f"{threshold:.2f}\t\t{f1:.4f}\t{precision:.4f}\t\t{recall:.4f}")

    print(f"\n=== Best threshold: {best_threshold:.2f} ===")
    print(f"  F1:        {best_f1:.4f}")
    print(f"  Precision: {best_precision:.4f}")
    print(f"  Recall:    {best_recall:.4f}")


if __name__ == "__main__":
    main()
