# PyTorch Tutorials: Deep Learning with PyTorch - A 60 Minute Blitz

A hands-on implementation of PyTorch's official beginner tutorials, progressing from tensor basics to training a complete image classifier.

## Overview

This repository contains Python implementations of the [PyTorch 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html) tutorials. The tutorials are designed to be followed sequentially, building foundational deep learning knowledge step by step.

## Tutorials

### 1. Tensors (pytorch1.py)

**Topic:** Tensor basics and operations

**Concepts Covered:**
- Creating tensors from data, NumPy arrays, and other tensors
- Tensor attributes (shape, dtype, device)
- Indexing, slicing, and concatenation
- Element-wise and matrix multiplication
- In-place operations
- NumPy bridge and GPU support

**Source:** [Tensor Tutorial](https://pytorch.org/tutorials/beginner/blitz/tensor_tutorial.html)

---

### 2. Autograd (pytorch2.py)

**Topic:** Automatic differentiation and a single training step

**Concepts Covered:**
- Forward pass through a pre-trained ResNet18
- Loss computation
- Backpropagation with `loss.backward()`
- Optimizer setup (SGD with momentum)
- Parameter updates

**Source:** [Autograd Tutorial](https://pytorch.org/tutorials/beginner/blitz/autograd_tutorial.html)

---

### 3. Neural Networks (pytorch3.py)

**Topic:** Building and training neural networks

**Concepts Covered:**
- Creating custom `nn.Module` classes
- Convolutional layers (`nn.Conv2d`)
- Fully connected layers (`nn.Linear`)
- Activation functions and pooling
- Loss functions (MSELoss)
- Gradient computation and weight updates

**Network Architecture:**
```
Input (1x32x32)
    ↓
Conv1 (1→6, 5x5) → ReLU → MaxPool
    ↓
Conv2 (6→16, 5x5) → ReLU → MaxPool
    ↓
FC1 (400→120) → ReLU
    ↓
FC2 (120→84) → ReLU
    ↓
FC3 (84→10)
    ↓
Output (10 classes)
```

**Source:** [Neural Networks Tutorial](https://pytorch.org/tutorials/beginner/blitz/neural_networks_tutorial.html)

---

### 4. Training a Classifier (pytorch4.py)

**Topic:** Complete end-to-end training on CIFAR-10

**Concepts Covered:**
- Loading and preprocessing datasets
- Data normalization and transformations
- DataLoader with batching
- Complete training loop (4 epochs)
- Model saving and loading
- Testing and per-class accuracy evaluation
- GPU/CPU device selection

**Dataset:** CIFAR-10 (10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)

**Training Configuration:**
- Loss: CrossEntropyLoss
- Optimizer: SGD (lr=0.001, momentum=0.9)
- Epochs: 4
- Batch size: 4

**Source:** [CIFAR-10 Tutorial](https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html)

## Requirements

```bash
pip install torch torchvision numpy matplotlib
```

**Dependencies:**
- Python 3.7+
- PyTorch
- TorchVision
- NumPy
- Matplotlib

## Usage

Run the tutorials in order:

```bash
# 1. Tensor basics
python pytorch1.py

# 2. Autograd and single training step
python pytorch2.py

# 3. Neural network construction
python pytorch3.py

# 4. Train CIFAR-10 classifier
python pytorch4.py
```

## Project Structure

```
Pytorch-tutorials/
├── README.md        # Project documentation
├── pytorch1.py      # Tensors tutorial
├── pytorch2.py      # Autograd tutorial
├── pytorch3.py      # Neural networks tutorial
└── pytorch4.py      # CIFAR-10 classifier training
```

## Generated Files

When running `pytorch4.py`:

| File | Description |
|------|-------------|
| `./data/` | CIFAR-10 dataset (downloaded automatically) |
| `./cifar_net.pth` | Trained model weights |

## Learning Path

```
pytorch1.py          pytorch2.py           pytorch3.py            pytorch4.py
    │                    │                     │                      │
    ▼                    ▼                     ▼                      ▼
 Tensors    →    Autograd/Gradients   →   Build Networks   →   Train Classifier
(foundations)   (backpropagation)      (architecture)       (end-to-end)
```

## References

- [PyTorch Official Tutorials](https://pytorch.org/tutorials/)
- [Deep Learning with PyTorch: A 60 Minute Blitz](https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html)
- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
