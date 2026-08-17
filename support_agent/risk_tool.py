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
        with open(MODEL_PATH, 'rb') as f:
            _model = pickle.load(f)
    return _model

def check_return_risk(order_features: dict) -> dict:
    """
    Evaluates the return risk of an order using the trained Random Forest model.
    
    Args:
        order_features (dict): A dictionary of features for the order.
    
    Returns:
        dict: A dictionary containing the predicted probability and the risk bucket.
    """
    model = load_model()
    
    # Convert the input dict to a DataFrame as expected by scikit-learn
    df = pd.DataFrame([order_features])
    
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
    # Example usage
    sample_order = {
        'Customer_Age': 30,
        'Customer_Location': 'Urban',
        'Customer_History_Returns': 2,
        'Product_Category': 'Electronics',
        'Product_Price': 15000,
        'Delivery_Time_Days': 3,
        'Payment_Method': 'Prepaid'
    }
    # Note: The actual features expected by the model depend on the preprocessing steps used in Part 1.
    # The dictionary provided should match those features exactly.
    try:
        print(check_return_risk(sample_order))
    except Exception as e:
        print(f"Error evaluating sample (likely missing or misformatted features): {e}")
