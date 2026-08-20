"""Train/test preprocessing with a leak-proof sklearn ColumnTransformer+Pipeline."""
import os
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "orders_dataset.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = os.path.join(ROOT_DIR, "orders_dataset.csv")
TARGET = "returned"

NUMERIC = [
    "price_inr",
    "discount_pct",
    "customer_tenure_days",
    "num_previous_orders",
    "num_previous_returns",
    "delivery_distance_km",
    "delivery_days",
    "is_weekend_order",
    "rating_given",
]
CATEGORICAL = ["product_category", "payment_method"]


def build_preprocessor() -> ColumnTransformer:
    """Return a fit-on-train-only preprocessing ColumnTransformer."""
    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, NUMERIC),
            ("cat", categorical_pipe, CATEGORICAL),
        ],
        remainder="drop",  # order_id and the target are dropped
    )


def load_and_split(test_size=0.2, random_state=42):
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC + CATEGORICAL]
    y = df[TARGET]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def get_fitted_data():
    """Convenience: split, fit preprocessor on train only, transform both."""
    X_train, X_test, y_train, y_test = load_and_split()
    preprocessor = build_preprocessor()
    X_train_t = preprocessor.fit_transform(X_train)
    X_test_t = preprocessor.transform(X_test)  # never fit on test
    return X_train_t, X_test_t, y_train, y_test, preprocessor


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, prep = get_fitted_data()
    feature_names = (
        NUMERIC
        + list(
            prep.named_transformers_["cat"]
            .named_steps["onehot"]
            .get_feature_names_out(CATEGORICAL)
        )
    )
    print(f"train: {X_train.shape}  test: {X_test.shape}")
    print(f"features: {len(feature_names)}  ->  {feature_names}")
    # Sanity: train means~0, std~1; no NaNs anywhere
    import numpy as np

    assert not np.isnan(X_train).any() and not np.isnan(X_test).any()
    print("no NaNs OK | scale check mean~0:", round(X_train[:, : len(NUMERIC)].mean(), 3))
