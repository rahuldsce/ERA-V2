PyTorch MNIST Image Classification This repository includes code for training a Convolutional Neural Network (CNN) to recognize handwritten digits from the MNIST dataset using PyTorch.

model.py:
The architecture of the CNN for image classification is defined in the model.py file. It introduces the Net class, outlining the layers and operations of the neural network. The file can be broken down as follows:

Importing Libraries: The initial section imports essential PyTorch libraries for defining neural networks (torch), neural network modules (torch.nn), and functional operations (torch.nn.functional).
Defining the Neural Network Architecture: The Net class delineates the CNN's structure, detailing convolutional layers, pooling layers, and fully connected layers using PyTorch's neural network module classes (nn.Conv2d, nn.MaxPool2d, nn.Linear, etc.).
Forward Pass: The forward method within the Net class specifies how input data undergoes processing through network layers to generate output predictions. It incorporates activation functions (F.relu) for convolutional layer outputs and reshapes the output tensor before passing it to fully connected layers.
utils.py:
The utils.py file comprises utility functions for training and testing the CNN model, aiding in data loading, training, testing, and accuracy calculation. The breakdown of the file is as follows:

Importing Libraries: Similar to model.py, the file initiates by importing necessary libraries such as PyTorch, torchvision, matplotlib, and tqdm.
Defining Utility Functions: The file encompasses various utility functions, including:
train: Used to train the CNN model with the training dataset, involving iterations over data batches, prediction computation, loss calculation, backpropagation, and model parameter updates.
test: Evaluates the trained model using the test dataset, calculating predictions, loss, and accuracy.
plot_train_metrics: Plots training and testing losses and accuracies, offering insight into the model's performance.
plot_sample_data: Plots sample data from the data loader, providing a visual understanding of the dataset. Additional utility functions may be added based on project requirements.
S5.ipynb:
The S5.ipynb Jupyter notebook functions as the primary script for training and evaluating the CNN model. It imports model.py and utils.py to define the model architecture and execute training and testing. The notebook can be broken down as follows:

Importing Libraries and Dependencies: The notebook starts by importing essential libraries and dependencies, including PyTorch, torchvision, matplotlib, tqdm, and torchsummary.
Defining Data Loaders and Data Visualization: Data loaders for the MNIST dataset are created using torchvision, and the data is transformed before passing it to the model. Visualization of the data aids in understanding its structure.
Defining Neural Network Model: The notebook loads the network model and sends it to the GPU for processing. A model summary is generated to provide information on the number of parameters and the model's structure.
Defining Hyperparameters: Hyperparameters such as batch size, learning rate, and the number of epochs are defined.
Training and Evaluating the Model: The notebook contains code cells for training the CNN model using the train function from utils.py and evaluating its performance using the test function. It iterates over multiple epochs, updating model parameters and monitoring training progress.
Plotting Training Metrics: After training, the notebook plots training loss, training accuracy, test loss, and test accuracy to visualize the model's performance across epochs.
