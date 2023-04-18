# MultiLayer Neural Network for Fruit Classification

This is a Python script that trains and evaluates a three-layer neural network on a fruit classification dataset. The neural network has two hidden layers with 256 and 64 nodes respectively. The input to the neural network is the first two principal components of the data obtained via PCA. The output layer of the neural network has three nodes, one for each fruit class.

## Dependencies

Python 3.x
numpy
scikit-learn
matplotlib
PIL

## Dataset

The dataset consists of 150 grayscale images of three types of fruits: Carambula, Lychee, and Pear. There are 50 images per class. The images are resized to 100x100 pixels and converted to grayscale before being fed into the neural network.

## Usage

1. Load and preprocess the dataset
2. Train the neural network
3. Evaluate the performance on validation data
4. Plot the loss curve
5. Plot the decision regions
6. Experiment with different batch sizes and hidden layer sizes

## Functions

### load_images(folder)

- Returns numpy arrays of flattened images and their corresponding labels
- Parameter: folder (str) - the directory where the images are stored

### preprocess_data(data_folder, pca_components)

- Returns preprocessed data for training and testing the neural network
- Parameters:
  - data_folder (str) - the directory where the data is stored
  - pca_components (int) - the number of principal components to keep after PCA

### one_hot_encode(y, num_classes)

- Returns one-hot encoded labels
- Parameters:
  - y (numpy array) - the labels to be encoded
  - num_classes (int) - the number of classes in the dataset

### sigmoid(x)

- Returns the sigmoid function applied to x
- Parameter: x (numpy array) - the input to the sigmoid function

### sigmoid_derivative(x)

- Returns the derivative of the sigmoid function applied to x
- Parameter: x (numpy array) - the input to the sigmoid function

### ThreeLayerNN(input_nodes, hidden_nodes_1, hidden_nodes_2, output_nodes)

- Initializes the neural network
- Parameters:
  - input_nodes (int) - the number of input nodes
  - hidden_nodes_1 (int) - the number of nodes in the first hidden layer
  - hidden_nodes_2 (int) - the number of nodes in the second hidden layer
  - output_nodes (int) - the number of output nodes

### softmax(x)

- Returns the softmax function applied to x
- Parameter: x (numpy array) - the input to the softmax function

### forward(X)

- Returns the output of the neural network for a given input
- Parameter: X (numpy array) - the input to the neural network

### backprop(X, y, output, learning_rate=0.001)

- Performs backpropagation and updates the weights and biases of the neural network
- Parameters:
  - X (numpy array) - the input to the neural network
  - y (numpy array) - the true labels for the input
  - output (numpy array) - the predicted output of the neural network
  - learning_rate (float) - the learning rate for the backpropagation

### cross_entropy_loss(y_true, y_pred)

- Returns the cross-entropy loss between the true labels and the predicted labels
- Parameters:
  - y_true (numpy array) - the true labels
  - y_pred (numpy array) - the predicted labels

### train(X, y, epochs=1000, learning_rate=0.001, batch_size=16)

- Trains the neural network on the input

### predict(X)

- Returns the predicted labels for a given input
- Parameter: X (numpy array) - the input to the neural network

### plot_loss()

- Plots the loss curve for the neural network

### plot_decision_regions(X, y, label)

- Plots the decision regions for the neural network on a 2D plane
- Parameters:
  - X (numpy array) - the input to the neural network
  - y (numpy array) - the true labels for the input
  - label (str) - the title for the plot

### custom_train_test_split(X, y, test_size=0.2, random_state=None)

- Performs a custom train-test split on the data
- Parameters:
  - X (numpy array) - the input data
  - y (numpy array) - the labels for the input data
  - test_size (float) - the percentage of data to use for testing
  - random_state (int) - the random seed for shuffling the data

### accuracy(y_true, y_pred)

- Returns the accuracy of the predicted labels compared to the true labels
- Parameters:
  - y_true (numpy array) - the true labels
  - y_pred (numpy array) - the predicted labels

## Conclusion

This Python script provides a basic implementation of a three-layer neural network for classification tasks. It is applied to a fruit classification dataset and shows the accuracy of the predicted labels compared to the true labels. It also provides the ability to experiment with different batch sizes and hidden layer sizes to see the effect on the loss curve.
