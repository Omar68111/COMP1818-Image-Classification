import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch
import torch.nn as nn

from pathlib import Path
from torchvision import datasets
from torchvision.models import resnet18, ResNet18_Weights
from torch.utils.data import DataLoader

# Select CPU or GPU
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Apply the same preprocessing used during ResNet training
weights = ResNet18_Weights.DEFAULT
transform = weights.transforms()

# Load the official CIFAR-10 test dataset
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

# Recreate the ResNet-18 architecture
model = resnet18(weights=None)

# Adapt the classifier to CIFAR-10's 10 classes
model.fc = nn.Linear(model.fc.in_features, 10)

# Load the best saved ResNet model
checkpoint_path = Path("models/resnet18_best.pth")

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

model = model.to(device)
model.eval()

print("Device:", device)
print("Test images:", len(test_dataset))
print("ResNet-18 checkpoint loaded successfully!")

# Evaluate ResNet-18 on the held-out test dataset
criterion = nn.CrossEntropyLoss()

total_loss = 0.0
total_correct = 0
total_images = 0
all_predictions = []
all_labels = []

with torch.no_grad():
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)

        # Generate predictions
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Count correct predictions
        predicted = outputs.argmax(dim=1)
        all_predictions.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_correct += (predicted == labels).sum().item()

        # Accumulate loss and image count
        batch_size = labels.size(0)
        total_images += batch_size
        total_loss += loss.item() * batch_size

# Calculate test accuracy and average loss
test_accuracy = 100 * total_correct / total_images
test_loss = total_loss / total_images

print("\nResNet-18 — Test Results")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct Predictions: {total_correct}/{total_images}")

# Generate ResNet-18 confusion matrix
cm = confusion_matrix(
    all_labels,
    all_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=test_dataset.classes
)

fig, ax = plt.subplots(figsize=(10, 8))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("ResNet-18 - CIFAR-10 Confusion Matrix")
plt.tight_layout()

plt.savefig("resnet_confusion_matrix.png", dpi=300)
plt.show()


