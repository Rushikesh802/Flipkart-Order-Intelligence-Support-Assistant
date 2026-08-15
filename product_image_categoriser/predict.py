import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

def predict_single_image(image_path, model_path="models/product_classifier.pt"):
    """
    Loads the trained ResNet-18 product classifier and predicts the class of a single image.
    
    Args:
        image_path (str): The path to the image file (e.g., .jpg or .png).
        model_path (str): The path to the saved PyTorch model weights (.pt).
        
    Returns:
        tuple: (predicted_class_name, confidence_score)
    """
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Define class names (based on FashionMNIST labels)
    classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    
    # 2. Define the transformation pipeline (matching the training pipeline)
    # FashionMNIST images are 1-channel Grayscale 28x28, but ResNet expects 3-channel 224x224 RGB.
    # We apply the same preprocessing that would be in your data loader.
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3), # Convert to 3 channel
        transforms.Resize((224, 224)),               # Resize for ResNet-18
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. Load the model architecture
    try:
        model = models.resnet18(weights=None)
    except AttributeError:
        model = models.resnet18(pretrained=False)
        
    # Replace the final fully connected layer
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, len(classes))
    
    # 4. Load the trained weights
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval() # Set to evaluation mode
    
    # 5. Load and preprocess the image
    img = Image.open(image_path)
    img_tensor = transform(img).unsqueeze(0).to(DEVICE) # Add batch dimension
    
    # 6. Predict
    with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)
        
    return classes[predicted_idx.item()], confidence.item()

# Example usage snippet:
if __name__ == "__main__":
    # example_image = "sample.jpg" 
    # pred_class, conf = predict_single_image(example_image)
    # print(f"Predicted Category: {pred_class} (Confidence: {conf:.2%})")
    pass
