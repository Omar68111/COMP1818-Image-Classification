
import torch

# Example predictions for four images
logits = torch.tensor([
    [2.5, 0.5, 0.1],
    [0.2, 1.8, 0.4],
    [0.5, 0.6, 1.9],
    [0.1, 2.1, 0.3]
])

# Correct labels
labels = torch.tensor([0, 1, 2, 0])

# Find the class with the highest score
predicted_classes = logits.argmax(dim=1)

# Compare predictions with actual labels
correct = (predicted_classes == labels).sum().item()

accuracy = correct / len(labels) * 100

print("Predicted classes:", predicted_classes.tolist())
print("Correct predictions:", correct)
print(f"Accuracy: {accuracy:.2f}%")
