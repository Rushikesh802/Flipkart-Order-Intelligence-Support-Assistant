import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score
from preprocess import load_and_split, build_preprocessor

def main():
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()
    pipe = Pipeline(
        steps=[("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42))]
    )
    pipe.fit(X_train, y_train)
    
    y_pred = pipe.predict(X_test)
    
    print(f"Overall Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Overall Recall: {recall_score(y_test, y_pred):.4f}")
    print("\n--- By Product Category ---")
    for cat in X_test['product_category'].unique():
        mask = X_test['product_category'] == cat
        if mask.sum() > 0:
            p = precision_score(y_test[mask], y_pred[mask], zero_division=0)
            r = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            print(f"{cat}: Precision={p:.4f}, Recall={r:.4f}, Support={mask.sum()}")
            
    print("\n--- By Payment Method ---")
    for method in X_test['payment_method'].unique():
        mask = X_test['payment_method'] == method
        if mask.sum() > 0:
            p = precision_score(y_test[mask], y_pred[mask], zero_division=0)
            r = recall_score(y_test[mask], y_pred[mask], zero_division=0)
            print(f"{method}: Precision={p:.4f}, Recall={r:.4f}, Support={mask.sum()}")

if __name__ == "__main__":
    main()
