import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import torch
from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from train_baseline import BaselineCNN

# Use the same image preprocessing as training
transform = transforms.ToTensor()

# Load the official CIFAR-10 test set
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

# Recreate the CNN architecture
model = BaselineCNN().to(device)

# Load the saved best model weights
checkpoint_path = Path("models/baseline_best.pth")

model.load_state_dict(
    torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
)

model.eval()

print("Device:", device)
print("Test images:", len(test_dataset))
print("Best baseline model loaded successfully!")



# Evaluate the model on the test dataset
criterion = torch.nn.CrossEntropyLoss()

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
        loss = criterion(outputs, labels)

        # Count correct classifications
        predicted = outputs.argmax(dim=1)
        all_predictions.extend(predicted.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())
        total_correct += (predicted == labels).sum().item()

        # Accumulate test loss
        batch_size = labels.size(0)
        total_images += batch_size
        total_loss += loss.item() * batch_size

# Calculate final test metrics
test_accuracy = 100 * total_correct / total_images
test_loss = total_loss / total_images

print("\nBaseline CNN — Test Results")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.2f}%")
print(f"Correct Predictions: {total_correct}/{total_images}")

# Generate the confusion matrix
class_names = test_dataset.classes

cm = confusion_matrix(
    all_labels,
    all_predictions
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=class_names
)

fig, ax = plt.subplots(figsize=(10, 8))

display.plot(
    ax=ax,
    cmap="Blues",
    xticks_rotation=45
)

plt.title("Baseline CNN - CIFAR-10 Confusion Matrix")
plt.tight_layout()

# Save the figure for the coursework report
plt.savefig("baseline_confusion_matrix.png", dpi=300)

plt.show()

