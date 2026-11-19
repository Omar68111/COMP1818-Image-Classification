
import torch
import torch.nn as nn
from pathlib import Path

from prepare_data import train_loader, val_loader


# -----------------------------------
# 1. Baseline CNN architecture
# -----------------------------------

class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 10)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from the previous batch
        optimizer.zero_grad()

        # Forward pass: generate predictions
        outputs = model(images)

        # Calculate prediction error
        loss = criterion(outputs, labels)

        # Calculate gradients
        loss.backward()

        # Update model weights
        optimizer.step()

        # Find the predicted class for each image
        predicted = outputs.argmax(dim=1)

        # Count correct predictions
        total_correct += (predicted == labels).sum().item()

        # Count images processed
        batch_size = labels.size(0)
        total_images += batch_size

        # Accumulate loss, accounting for batch size
        total_loss += loss.item() * batch_size



    # Calculate average training loss and accuracy
    average_loss = total_loss / total_images
    accuracy = 100 * total_correct / total_images

    return average_loss, accuracy



def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_images = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            # Generate predictions
            outputs = model(images)

            # Calculate validation loss
            loss = criterion(outputs, labels)

            # Count correct predictions
            predicted = outputs.argmax(dim=1)
            total_correct += (predicted == labels).sum().item()

            # Accumulate loss and image count
            batch_size = labels.size(0)
            total_images += batch_size
            total_loss += loss.item() * batch_size

    average_loss = total_loss / total_images
    accuracy = 100 * total_correct / total_images

    return average_loss, accuracy


# -----------------------------------
# 2. Full training and validation
# -----------------------------------

def main():
    torch.manual_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    model = BaselineCNN().to(device)

    # Load the previously trained baseline
    checkpoint_path = Path("models/baseline_best.pth")

    if checkpoint_path.exists():
        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True
            )
        )
        print("Loaded previously trained baseline model")
    else:
        print("No saved checkpoint found")

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.0003
    )

    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))



    # Train and validate for 5 epochs
    num_epochs = 5

    print("\nStarting 5-epoch baseline training...")


    # Prepare model checkpoint directory
    checkpoint_dir = Path("models")
    checkpoint_dir.mkdir(exist_ok=True)

    best_val_accuracy = 64.69

    for epoch in range(num_epochs):

        train_loss, train_accuracy = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )

        val_loss, val_accuracy = validate_one_epoch(
            model, val_loader, criterion, device
        )

        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print(f"Training Loss: {train_loss:.4f}")
        print(f"Training Accuracy: {train_accuracy:.2f}%")
        print(f"Validation Loss: {val_loss:.4f}")
        print(f"Validation Accuracy: {val_accuracy:.2f}%")








        # Save the best model based on validation accuracy
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy

            torch.save(
                model.state_dict(),
                checkpoint_dir / "baseline_best.pth"
            )

            print("New best model saved!")



if __name__ == "__main__":
    main()
