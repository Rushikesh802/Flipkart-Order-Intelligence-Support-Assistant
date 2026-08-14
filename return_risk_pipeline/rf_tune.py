"""Tune a RandomForestClassifier through the same leak-proof pipeline.

GridSearchCV over n_estimators in [100, 200] and max_depth in [6, 10, None],
scored on roc_auc with 5-fold StratifiedKFold cross-validation.
Reports best params, best CV ROC-AUC, and held-out test ROC-AUC for the
winning configuration.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from preprocess import load_and_split, build_preprocessor


def main():
    X_train, X_test, y_train, y_test = load_and_split()

    # Wrap the SAME preprocessor from preprocess.py + the RF classifier
    preprocessor = build_preprocessor()
    pipe = Pipeline(
        steps=[("prep", preprocessor), ("clf", RandomForestClassifier(class_weight="balanced", random_state=42))]
    )

    param_grid = {
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [6, 10, None],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    best = grid.best_estimator_
    best_proba = best.predict_proba(X_test)[:, 1]
    test_auc = roc_auc_score(y_test, best_proba)

    print("=== Random Forest GridSearchCV (roc_auc, 5-fold StratifiedKFold) ===")
    print(f"Best params: {grid.best_params_}")
    print(f"Best CV ROC-AUC: {grid.best_score_:.4f}")
    print(f"Held-out test ROC-AUC: {test_auc:.4f}")
    # Show the full CV results table for transparency
    print("\nCV results per param combo (mean ROC-AUC):")
    for mean, params in zip(grid.cv_results_["mean_test_score"], grid.cv_results_["params"]):
        print(f"  {params} -> {mean:.4f}")


if __name__ == "__main__":
    main()
