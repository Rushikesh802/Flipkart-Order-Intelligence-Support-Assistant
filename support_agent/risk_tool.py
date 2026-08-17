import joblib
import pickle
import pandas as pd
import os

# Best threshold found during Random Forest tuning
T_STAR_RF = 0.44

# Path to the model
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'return_risk_model.pkl')

# Load the model lazily or at initialization
_model = None

def load_model():
    global _model
    if _model is None:
        try:
            _model = joblib.load(MODEL_PATH)
        except Exception:
            with open(MODEL_PATH, 'rb') as f:
                _model = pickle.load(f)
    return _model

DEFAULT_ORDER_FEATURES = {
    'price_inr': 1500.0,
    'discount_pct': 10.0,
    'customer_tenure_days': 180,
    'num_previous_orders': 5,
    'num_previous_returns': 1,
    'delivery_distance_km': 15.0,
    'delivery_days': 3,
    'is_weekend_order': 0,
    'rating_given': 4.0,
    'product_category': 'electronics',
    'payment_method': 'prepaid'
}

def check_return_risk(order_features: dict) -> dict:
    """
    Evaluates the return risk of an order using the trained Random Forest model.
    
    Args:
        order_features (dict): A dictionary of features for the order.
    
    Returns:
        dict: A dictionary containing the predicted probability and the risk bucket.
    """
    model = load_model()
    
    # Merge input with defaults for any missing features
    features = DEFAULT_ORDER_FEATURES.copy()
    for k, v in order_features.items():
        # Handle lowercase/uppercase key differences
        norm_k = k.lower().strip()
        matched = False
        for dk in DEFAULT_ORDER_FEATURES:
            if norm_k == dk.lower():
                features[dk] = v
                matched = True
                break
        if not matched:
            features[k] = v
            
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Predict the probability of return (class 1)
    probability = model.predict_proba(df)[0][1]
    
    # Determine the risk bucket based on t*_rf
    if probability < T_STAR_RF:
        risk_bucket = "Low"
    elif probability >= T_STAR_RF + 0.15:
        risk_bucket = "High"
    else:
        risk_bucket = "Medium"
        
    return {
        "predicted_probability": float(probability),
        "risk_bucket": risk_bucket
    }

if __name__ == "__main__":
    sample_order = {
        'price_inr': 15000,
        'discount_pct': 15,
        'customer_tenure_days': 365,
        'num_previous_orders': 10,
        'num_previous_returns': 2,
        'delivery_distance_km': 25,
        'delivery_days': 3,
        'is_weekend_order': 0,
        'rating_given': 4.5,
        'product_category': 'electronics',
        'payment_method': 'prepaid'
    }
    print(check_return_risk(sample_order))
