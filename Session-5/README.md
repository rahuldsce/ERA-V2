# PyTorch MNIST Image Classification

This repository includes code to train a convolutional neural network (CNN) for classifying handwritten digits from the MNIST dataset using PyTorch.

## Code Files

### model.py

The `model.py` file outlines the CNN architecture for image classification. It introduces the `Net` class, defining the neural network's layers and operations. The breakdown of the file is as follows:

- **Library Imports**: The file starts by importing essential PyTorch libraries for neural network definition (`torch`), neural network modules (`torch.nn`), and functional operations (`torch.nn.functional`).

- **Neural Network Structure**: Within the `Net` class, the CNN structure is defined, specifying convolutional layers, pooling layers, and fully connected layers. PyTorch's neural network module classes (`nn.Conv2d`, `nn.MaxPool2d`, `nn.Linear`, etc.) are used for each layer.

- **Forward Pass**: The `forward` method in the `Net` class outlines how input data traverses the network layers to produce output predictions. Activation functions (`F.relu`) are applied to convolutional layer outputs, and the output tensor is reshaped before reaching fully connected layers.

### utils.py

The `utils.py` file hosts utility functions for training and testing the CNN model. These functions streamline data loading, training, testing, and accuracy calculation.
