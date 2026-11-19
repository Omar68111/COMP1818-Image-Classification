
# COMP1818 Experiment Results

## Dataset
CIFAR-10
Training images: 40,000
Validation images: 10,000
Test images: 10,000
Random seed: 42

## Final Results

| Metric | Baseline CNN | ResNet-18 |
|---|---:|---:|
| Test accuracy | 69.32% | 85.15% |
| Test loss | 0.8908 | 0.4209 |
| Best validation accuracy | 68.99% | 85.68% |
| Correct test predictions | 6932 | 8515 |
| Incorrect test predictions | 3068 | 1485 |

## Baseline CNN
Architecture: Two convolutional layers
Training: 5 initial epochs + 5 additional epochs
Initial learning rate: 0.001
Additional training learning rate: 0.0003
Best checkpoint: models/baseline_best.pth

## ResNet-18
Architecture: ResNet-18
Initialisation: ImageNet pretrained weights
Training: Frozen backbone, replacement classifier
Epochs: 3
Learning rate: 0.001
Trainable parameters: 5130
Best checkpoint: models/resnet18_best.pth

## Observations
- ResNet improved test accuracy by 15.83 percentage points.
- Both models performed worse on some visually similar animal categories.
- ResNet still confused 192 dogs with cats.
- The comparison includes different initialisation and preprocessing,
  so the gain cannot be attributed to residual connections alone.
   