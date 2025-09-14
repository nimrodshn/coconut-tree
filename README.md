# coconut-tree

A simple set of experiments with Meta's "COCONUT" paradigm (Continuous "chain-of-thought"). This project implements both a baseline Vision Transformer for MNIST classification and an enhanced version using the COCONUT continuous reasoning approach.

## Overview

The COCONUT (Continuous Chain-of-Thought) paradigm allows models to perform reasoning in continuous latent space rather than through discrete token sequences. This eliminates the tokenization bottleneck and enables more fluid reasoning processes.

## Project Structure

- `train_mnist_simple.py` - Baseline Vision Transformer implementation and training
- `mnist_coconut.py` - COCONUT architecture adapted for MNIST classification
- `train_mnist_coconut.py` - Training script for COCONUT model with comparison capabilities
- `data/` - Directory for MNIST dataset files
- `mnist_attention_checkpoint/` - Saved baseline Vision Transformer model

## Features

### Simple MNIST Classifier (Vision Transformer)

A standard Vision Transformer implementation for MNIST digit classification:
- Patch-based image encoding (4x4 patches)
- Multi-head attention mechanism
- Transformer blocks with layer normalization
- Classification head for 10-digit prediction

### COCONUT MNIST Classifier

An enhanced version that implements continuous reasoning:
- **Continuous Latent Reasoning**: Uses hidden states directly as embeddings without tokenization
- **Autoregressive Reasoning Loop**: Iteratively refines latent thoughts
- **Multi-step Classification**: Generates predictions through reasoning steps
- **Flexible Architecture**: Can operate in both standard and reasoning modes

## Usage

### 1. Prepare MNIST Data

Download MNIST dataset files and place them in the `data/mnist/` directory:
- `train-images.idx3-ubyte`
- `train-labels.idx1-ubyte`
- `t10k-images.idx3-ubyte`
- `t10k-labels.idx1-ubyte`

### 2. Train Baseline Vision Transformer

```bash
python train_mnist_simple.py --data_path data --save_path mnist_attention_checkpoint --epochs 10
```

### 3. Train COCONUT Model

```bash
# Train with COCONUT reasoning (default)
python train_mnist_coconut.py --model_path mnist_attention_checkpoint --data_path data --save_path mnist_coconut_checkpoint --epochs 5 --reasoning_steps 10

# Train regular model for comparison
python train_mnist_coconut.py --model_path mnist_attention_checkpoint --data_path data --save_path mnist_regular_checkpoint --epochs 5 --use_regular
```

### 4. Test COCONUT Model

```python
from mnist_coconut import create_mnist_coconut_from_pretrained
import torch

# Load trained model
model, config = create_mnist_coconut_from_pretrained("mnist_coconut_checkpoint")

# Generate predictions with reasoning
images = torch.randn(1, 1, 28, 28)  # Dummy image
predictions, reasoning_logits = model.generate(images, max_reasoning_steps=10)
```

## Arguments

### train_mnist_simple.py
- `--data_path`: Path to MNIST data directory (default: "data")
- `--save_path`: Path to save model (default: "mnist_attention_checkpoint")
- `--epochs`: Number of training epochs (default: 10)
- `--batch_size`: Batch size (default: 64)
- `--lr`: Learning rate (default: 1e-3)

### train_mnist_coconut.py
- `--model_path`: Path to pretrained Vision Transformer (default: "mnist_attention_checkpoint")
- `--data_path`: Path to MNIST data (default: "data")
- `--save_path`: Path to save COCONUT model (default: "mnist_coconut_checkpoint")
- `--epochs`: Number of epochs (default: 3)
- `--batch_size`: Batch size (default: 32)
- `--lr`: Learning rate (default: 1e-4)
- `--reasoning_steps`: Number of reasoning steps (default: 10)
- `--use_coconut`: Use COCONUT reasoning (default: True)
- `--use_regular`: Use regular training instead of COCONUT

## Key Differences: Standard vs COCONUT

| Feature | Standard ViT | COCONUT ViT |
|---------|-------------|-------------|
| Reasoning | Single forward pass | Multi-step continuous reasoning |
| Latent Updates | Static embeddings | Dynamic latent thought evolution |
| Output | Direct classification | Reasoning sequence + classification |
| Training | Standard cross-entropy | Reasoning-aware loss |
| Inference | Fast single pass | Interpretable reasoning steps |

## Dependencies

- PyTorch
- NumPy
- tqdm (for progress bars)
- Standard Python libraries (os, struct, argparse)

## Model Architecture Details

### Vision Transformer Components
- **Patch Embedding**: Converts 28x28 images into 7x7 grid of 4x4 patches
- **Multi-Head Attention**: 4 attention heads with 64-dimensional embeddings
- **Transformer Blocks**: 4 layers with layer normalization and GELU activation
- **Classification Head**: Linear layer for 10-class digit prediction

### COCONUT Enhancements
- **Special Tokens**: Latent, start, and end tokens for reasoning sequences
- **Continuous Reasoning**: Hidden states directly become next embeddings
- **Latent Update Module**: Optional transformation for thought refinement
- **Extended Positional Embeddings**: Support for variable-length reasoning

## Experimental Results

The COCONUT model demonstrates the ability to perform multi-step reasoning for MNIST classification, providing interpretable reasoning steps while maintaining competitive accuracy with the baseline Vision Transformer.