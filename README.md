
# COMP1818 CIFAR-10 Image Classification

## Project Overview
This project investigates image classification using a
baseline Convolutional Neural Network (CNN) and a
pretrained ResNet-18 model.

## Research Paper
Deep Residual Learning for Image Recognition
He et al. (2016)

## Dataset
CIFAR-10

Training: 40,000 images
Validation: 10,000 images
Testing: 10,000 images

## Models
1. Baseline CNN trained from scratch
2. Pretrained ResNet-18 with a frozen backbone

## Installation
Create a Python virtual environment and install
the dependencies listed in requirements.txt.

## Running the Experiments
Data preparation:
python prepare_data.py

Baseline training:
python train_baseline.py

Baseline evaluation:
python evaluate_baseline.py

ResNet training:
python train_resnet.py

ResNet evaluation:
python evaluate_resnet.py

## Results
See experiment_results.md for the measured results.

## Model Checkpoints
models/baseline_best.pth
models/resnet18_best.pth
