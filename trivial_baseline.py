
from torchvision import datasets

# Load official CIFAR-10 test labels
test_dataset = datasets.CIFAR10(
    root="./data",
    train=False,
    download=True
)

actual_labels = test_dataset.targets

print("Number of test images:", len(actual_labels))
print("Class names:", test_dataset.classes)

# Trivial classifier: always predict class 0 (airplane)
predicted_labels = [0] * len(actual_labels)

# Compare predictions against actual labels
correct_predictions = sum(
    predicted == actual
    for predicted, actual in zip(predicted_labels, actual_labels)
)

# Calculate accuracy
accuracy = 100 * correct_predictions / len(actual_labels)

print("\nTrivial Baseline — Test Results")
print(f"Test Accuracy: {accuracy:.2f}%")
print(f"Correct Predictions: {correct_predictions}/{len(actual_labels)}")

