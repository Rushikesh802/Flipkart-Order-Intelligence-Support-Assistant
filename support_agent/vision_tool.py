import os
import sys

# Add root directory to path to import the categoriser module
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from product_image_categoriser.predict import predict_single_image

def classify_product_image(image_path: str) -> dict:
    """
    Loads Part 2's saved classifier (product_classifier.pt) and returns the 
    predicted category label plus the model's confidence for that prediction.
    
    Args:
        image_path (str): The absolute or relative path to the image file.
        
    Returns:
        dict: A dictionary containing the 'predicted_category' and 'confidence'.
    """
    model_path = os.path.join(ROOT_DIR, "models", "product_classifier.pt")
    
    # predict_single_image handles preprocessing and PyTorch model loading
    pred_class, conf = predict_single_image(image_path, model_path=model_path)
    
    return {
        "predicted_category": pred_class,
        "confidence": float(conf)
    }

if __name__ == "__main__":
    import glob
    
    # Run against the real .png files exported to data/sample_images/
    sample_images_dir = os.path.join(ROOT_DIR, "data", "sample_images")
    png_files = glob.glob(os.path.join(sample_images_dir, "*.png"))
    
    print("Testing classify_product_image on data/sample_images/:")
    for file in png_files:
        result = classify_product_image(file)
        print(f"File: {os.path.basename(file)}")
        print(f"  Prediction: {result['predicted_category']}")
        print(f"  Confidence: {result['confidence']:.4f}\n")
