import torch
import torch.nn as nn
import torchvision.models as models
from data_loader import get_dataloaders
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def evaluate():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    # 1. Load data
    print("Loading test dataset...")
    _, _, test_loader_img, classes = get_dataloaders(batch_size=64)

    # 2. Load the full model (ResNet-18 + custom head)
    print("Loading the full product classifier model...")
    try:
        model = models.resnet18(weights=None)
    except AttributeError:
        model = models.resnet18(pretrained=False)
        
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 10)
    
    # Load the full state dict
    model.load_state_dict(torch.load("models/product_classifier.pt", map_location=DEVICE))
    model = model.to(DEVICE)
    model.eval()

    # 4. Evaluate
    print("Evaluating on test set...")
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader_img):
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            
            if (i+1) % 50 == 0:
                print(f"  Processed {i+1} batches...")

    # Calculate metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    acc = np.mean(all_preds == all_targets) * 100
    print(f"\nFinal Test Accuracy: {acc:.2f}%")
    
    print("\nConfusion Matrix:")
    cm = confusion_matrix(all_targets, all_preds)
    print(cm)
    
    print("\nPer-class Precision/Recall:")
    target_names = classes if classes else [str(i) for i in range(10)]
    print(classification_report(all_targets, all_preds, target_names=target_names))

if __name__ == "__main__":
    evaluate()
