from train_baseline import validate_one_epoch
from pathlib import Path
import time
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


def train_frozen_resnet_one_epoch(model, loader, criterion, optimizer, device):
    # Keep the frozen backbone and BatchNorm statistics unchanged
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation updates only the classifier
        loss.backward()
        optimizer.step()

        predicted = outputs.argmax(dim=1)
        total_correct += (predicted == labels).sum().item()

        batch_size = labels.size(0)
        total_images += batch_size
        total_loss += loss.item() * batch_size

    average_loss = total_loss / total_images
    accuracy = 100 * total_correct / total_images

    return average_loss, accuracy


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet-18
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# Freeze pretrained feature layers
for parameter in model.parameters():
    parameter.requires_grad = False

# Replace ImageNet classifier with 10 CIFAR-10 classes
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, 10)

model = model.to(device)

print("Device:", device)
print("Classifier:", model.fc)
print(
    "Trainable parameters:",
    sum(p.numel() for p in model.parameters() if p.requires_grad)
)

# Preprocessing for pretrained ResNet-18
transform = weights.transforms()

# Load CIFAR-10 using ResNet preprocessing
full_train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Reproduce the baseline's 40,000 / 10,000 split

# Reproduce the baseline's random_split indices
from torch.utils.data import random_split

split_generator = torch.Generator().manual_seed(42)

train_split, val_split = random_split(
    range(len(full_train_dataset)),
    [40000, 10000],
    generator=split_generator
)

train_indices = train_split.indices
val_indices = val_split.indices


train_dataset = Subset(full_train_dataset, train_indices)
val_dataset = Subset(full_train_dataset, val_indices)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))

images, labels = next(iter(train_loader))
print("Image batch shape:", images.shape)
print("Labels shape:", labels.shape)

# Check ResNet-18 inference speed on CPU
model.eval()

start_time = time.perf_counter()

with torch.no_grad():
    outputs = model(images.to(device))

elapsed_time = time.perf_counter() - start_time


# ResNet classifier training configuration
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.fc.parameters(),
    lr=0.001
)

num_epochs = 3

print("\nResNet training configuration:")
print("Epochs:", num_epochs)
print("Learning rate:", optimizer.param_groups[0]["lr"])
print("Training batches:", len(train_loader))
print("Validation batches:", len(val_loader))

# Save the strongest ResNet classifier
checkpoint_dir = Path("models")
checkpoint_dir.mkdir(exist_ok=True)

best_val_accuracy = 0.0

for epoch in range(num_epochs):
    train_loss, train_accuracy = train_frozen_resnet_one_epoch(
        model, train_loader, criterion, optimizer, device
    )

    val_loss, val_accuracy = validate_one_epoch(
        model, val_loader, criterion, device
    )

    print(f"\nResNet Epoch {epoch + 1}/{num_epochs}")
    print(f"Training Loss: {train_loss:.4f}")
    print(f"Training Accuracy: {train_accuracy:.2f}%")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.2f}%")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(
            model.state_dict(),
            checkpoint_dir / "resnet18_best.pth"
        )
        print("New best ResNet model saved!")



print("\nResNet-18 Speed Test")
print("Output shape:", outputs.shape)
print(f"Time for one batch: {elapsed_time:.2f} seconds")

