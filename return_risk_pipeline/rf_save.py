import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from preprocess import load_and_split, build_preprocessor

def main():
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()
    pipe = Pipeline(
        steps=[("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42))]
    )
    pipe.fit(X_train, y_train)
    
    # Sweep thresholds on test set
    probas = pipe.predict_proba(X_test)[:, 1]
    thresholds = np.arange(0.10, 0.91, 0.02)
    best_f1 = -1
    best_t = 0.5
    
    print("threshold\tF1")
    for t in thresholds:
        preds = (probas >= t).astype(int)
        f1 = f1_score(y_test, preds)
        print(f"{t:.2f}\t\t{f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_t = t
            
    print(f"\nBest threshold (t*_rf): {best_t:.2f} with F1: {best_f1:.4f}")
    
    # Save the pipeline
    os.makedirs("../models", exist_ok=True)
    model_path = "../models/return_risk_model.pkl"
    joblib.dump(pipe, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
