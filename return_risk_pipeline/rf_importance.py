import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from preprocess import load_and_split, build_preprocessor

def main():
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()
    pipe = Pipeline(
        steps=[("prep", preprocessor), ("clf", RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42))]
    )
    pipe.fit(X_train, y_train)
    
    feature_names = preprocessor.get_feature_names_out()
    rf = pipe.named_steps["clf"]
    gini_importances = rf.feature_importances_
    
    print("=== Gini (Impurity-based) Importance ===")
    gini_indices = np.argsort(gini_importances)[::-1]
    for i in range(5):
        print(f"{feature_names[gini_indices[i]]}: {gini_importances[gini_indices[i]]:.4f}")
        
    print("\n=== Permutation Importance (Test Set) ===")
    X_test_prep = preprocessor.transform(X_test)
    result = permutation_importance(rf, X_test_prep, y_test, scoring="roc_auc", n_repeats=10, random_state=42, n_jobs=-1)
    perm_importances = result.importances_mean
    perm_indices = np.argsort(perm_importances)[::-1]
    for i in range(5):
        print(f"{feature_names[perm_indices[i]]}: {perm_importances[perm_indices[i]]:.4f}")

if __name__ == "__main__":
    main()
