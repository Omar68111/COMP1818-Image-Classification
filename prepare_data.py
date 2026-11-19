
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from collections import Counter
import matplotlib.pyplot as plt

# -------------------------------
# 1. Load CIFAR-10
# -------------------------------

transform = transforms.ToTensor()

train_dataset = datasets.CIFAR10(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# -------------------------------
# 2. Reproducible train/val split
# -------------------------------

generator = torch.Generator().manual_seed(42)

train_subset, val_subset = random_split(
    train_dataset,
    [40000, 10000],
    generator=generator
)

# -------------------------------
# 3. Create DataLoaders
# -------------------------------

train_loader = DataLoader(
    train_subset,
    batch_size=64,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_subset,
    batch_size=64,
    shuffle=False,
    num_workers=0
)

# -------------------------------
# 4. Dataset inspection
# Only runs when this file is run
# directly, not when imported.
# -------------------------------

if __name__ == "__main__":

    print("Training images:", len(train_dataset))
    print("Classes:", train_dataset.classes)

    print("\nDataset Split:")
    print("Training:", len(train_subset))
    print("Validation:", len(val_subset))

    train_labels = [
        train_dataset.targets[i]
        for i in train_subset.indices
    ]

    val_labels = [
        train_dataset.targets[i]
        for i in val_subset.indices
    ]

    train_counts = Counter(train_labels)
    val_counts = Counter(val_labels)

    print("\nClass Distribution After Splitting:")
    print(f"{'Class':<15} {'Training':<12} {'Validation'}")

    for class_id, class_name in enumerate(train_dataset.classes):
        print(
            f"{class_name:<15} "
            f"{train_counts[class_id]:<12} "
            f"{val_counts[class_id]}"
        )

    images, labels = next(iter(train_loader))

    print("\nDataLoader Test:")
    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)
    print("Training batches:", len(train_loader))
    print("Validation batches:", len(val_loader))
