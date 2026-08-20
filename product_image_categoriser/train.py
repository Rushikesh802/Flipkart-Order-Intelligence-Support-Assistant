import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from torch.utils.data import DataLoader, TensorDataset
import time
from data_loader import get_dataloaders

# ==========================================
# Configuration Details (as requested)
# ==========================================
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS = 10
OPTIMIZER_NAME = "Adam"
# ==========================================

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_features(model, dataloader, device):
    features = []
    labels = []
    
    # Ensure backbone is in eval mode (e.g., for BatchNorm/Dropout layers)
    model.eval()
    
    with torch.no_grad(): # No gradients needed for feature extraction
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(device)
            # Forward pass through the backbone to get features
            out = model(inputs)
            features.append(out.cpu())
            labels.append(targets)
            if (i+1) % 100 == 0:
                print(f"  Processed {i+1} batches...")
            
    return torch.cat(features), torch.cat(labels)

def main():
    print("==========================================")
    print(f"Using device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Optimizer: {OPTIMIZER_NAME}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Epochs: {EPOCHS}")
    print("==========================================\n")
    
    # 1. Load data using the provided data_loader.py
    print("Loading datasets (this may take a moment to download if not present)...")
    train_loader_img, val_loader_img, test_loader_img, classes = get_dataloaders(batch_size=BATCH_SIZE)
    
    # 2. Load pretrained model (ResNet-18)
    print("\nLoading pretrained ResNet-18 backbone...")
    try:
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    except AttributeError:
        # Fallback for older PyTorch versions
        backbone = models.resnet18(pretrained=True)
        
    # 3. Freeze backbone & prepare for feature extraction
    # By replacing the final fully connected layer with an Identity layer,
    # the model will output the raw feature vectors directly.
    num_features = backbone.fc.in_features
    backbone.fc = nn.Identity()
    backbone = backbone.to(DEVICE)
    
    # 4. Extract and cache features (The Speed Tip)
    # Since the backbone is frozen during feature extraction, we run it once over 
    # every image to extract and cache its output features. This turns a long CPU training loop
    # into a quick single pass followed by a near-instant head-only training step.
    print("\nExtracting and caching features from training set...")
    start_time = time.time()
    train_features, train_labels = extract_features(backbone, train_loader_img, DEVICE)
    
    print("\nExtracting and caching features from validation set...")
    val_features, val_labels = extract_features(backbone, val_loader_img, DEVICE)
    
    print("\nExtracting and caching features from test set...")
    test_features, test_labels = extract_features(backbone, test_loader_img, DEVICE)
    
    extraction_time = time.time() - start_time
    print(f"\nFeature extraction completed in {extraction_time:.2f} seconds.")
    
    # Create new dataloaders for the cached features
    train_dataset = TensorDataset(train_features, train_labels)
    val_dataset = TensorDataset(val_features, val_labels)
    test_dataset = TensorDataset(test_features, test_labels)
    
    # We can use a large batch size here if we want, but sticking to BATCH_SIZE
    train_loader_feat = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader_feat = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader_feat = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # 5. Create new classifier head
    # 10 output classes for FashionMNIST
    classifier_head = nn.Linear(num_features, 10).to(DEVICE)
    
    # 6. Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier_head.parameters(), lr=LEARNING_RATE)
    
    # 7. Train ONLY the classifier head on the cached features
    print("\nTraining the classifier head on cached features...")
    training_start = time.time()
    for epoch in range(EPOCHS):
        classifier_head.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, targets in train_loader_feat:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = classifier_head(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total
        
        # Validation pass
        classifier_head.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, targets in val_loader_feat:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                outputs = classifier_head(inputs)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
                
        val_loss = val_loss / val_total
        val_acc = 100. * val_correct / val_total
        
        print(f"Epoch {epoch+1}/{EPOCHS} | "
              f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
              
    print(f"Head training completed in {time.time() - training_start:.2f} seconds.")
              
    # 8. Final Evaluation on test set
    print("\nEvaluating on test set...")
    classifier_head.eval()
    test_correct = 0
    test_total = 0
    with torch.no_grad():
        for inputs, targets in test_loader_feat:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = classifier_head(inputs)
            _, predicted = outputs.max(1)
            test_total += targets.size(0)
            test_correct += predicted.eq(targets).sum().item()
            
    print(f"Test Accuracy: {100. * test_correct / test_total:.2f}%")
    
    # Save the trained head weights
    torch.save(classifier_head.state_dict(), "classifier_head.pth")
    print("\nSaved classifier head weights to classifier_head.pth")

    # Assemble and save the full model state dict to models/product_classifier.pt
    import os
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, "product_classifier.pt")

    try:
        full_model = models.resnet18(weights=None)
    except AttributeError:
        full_model = models.resnet18(pretrained=False)
    full_model.fc = nn.Linear(num_features, 10)
    full_sd = backbone.state_dict()
    full_model_sd = full_model.state_dict()
    for k in full_sd:
        if k in full_model_sd and not k.startswith("fc."):
            full_model_sd[k] = full_sd[k]
    full_model_sd["fc.weight"] = classifier_head.weight.data.cpu()
    full_model_sd["fc.bias"] = classifier_head.bias.data.cpu()
    full_model.load_state_dict(full_model_sd)
    torch.save(full_model.state_dict(), model_save_path)
    print(f"Saved full model artifact to {model_save_path}")

if __name__ == '__main__':
    main()
