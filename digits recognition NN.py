#Importing libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv('digits.csv')


# Converting the data into a matrix, i.e. a numpy array
data = np.array(data)
# Storing the dimensions of the data as m and n
m, n = data.shape
# Shuffling the data
np.random.shuffle(data)

# Setting aside 1000 datapoints to use to test final performance
data_test = data[0:1000].T
Y_test = data_test[0]
X_test = data_test[1:n]
X_test = X_test / 255.

# Setting aside the rest of the datapoints to train the network
data_train = data[1000:m].T
Y_train = data_train[0]
X_train = data_train[1:n]
X_train = X_train / 255.
_,m_train = X_train.shape


# Initializing the weights and biases to random values for the first iteration
def init_params():
    W1 = np.random.rand(16, 784) - 0.5
    b1 = np.random.rand(16, 1) - 0.5
    W2 = np.random.rand(10, 16) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


# The sigmoid activation function
def sigmoid(Z): 
    return 1 / (1 + np.exp(-Z))

# The softmax activation function
def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z))
    return A

# Forward propagation
def forward_propagation(W1, b1, W2, b2, X):
      # First layer
    Z1 = W1.dot(X) + b1
    A1 = sigmoid(Z1)
    # Second layer
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2



# The derivative of the sigmoid function
def sigmoid_deriv(Z):
    return sigmoid(Z) * (1- sigmoid(Z))


# One-hot encoding the correct labels
def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, Y.max() + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y


 #Backward propagation
def backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y):
    one_hot_Y = one_hot(Y)
    # Second layer
    dZ2 = A2 - one_hot_Y  
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    # First layer
    dZ1 = W2.T.dot(dZ2) * sigmoid_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

# Updating parameters
def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    # First layer
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1 
    # Second layer
    W2 = W2 - alpha * dW2  
    b2 = b2 - alpha * db2    
    return W1, b1, W2, b2

print(one_hot(Y_train).shape)



def get_accuracy(predictions, Y):
    print(predictions, Y)
    return np.sum(predictions == Y) / Y.size


def get_predictions(A2):
    return np.argmax(A2, 0) #returns max element indices i.e A2=[0.1,0.9,0.3,...,0.4],[0.2,0.95,0.3,..0.1],[],... --> [2,2,..]  (dim of A2 10x4100)

# Training the network
def gradient_descent(X, Y, iterations, alpha):
    W1, b1, W2, b2 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_propagation(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_propagation(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)
        if (i % 10 == 0):
            print("Iteration: ", i)
            print("Accuracy: ", get_accuracy(get_predictions(A2), Y))
    return W1, b1, W2, b2



# Runs the previous code
W1, b1, W2, b2 = gradient_descent(X_train, Y_train, 101, 0.1)


# Making predictions about image labels using forward propagations of the network
def make_predictions(X, W1, b1, W2, b2):
    _, _, _, A2 = forward_propagation(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    return predictions

# Testing predictions made using the network and plotting them
def test_prediction(index, W1, b1, W2, b2):
    current_image = X_train[:, index, None]
    prediction = make_predictions(X_train[:, index, None], W1, b1, W2, b2)
    label = Y_train[index]
    print("Prediction: ", prediction)
    print("Label: ", label)
    
    current_image = current_image.reshape((28, 28)) * 255
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()
    
    
    
    
# Testing some sample predictions
test_prediction(0, W1, b1, W2, b2)
test_prediction(1, W1, b1, W2, b2)
test_prediction(2, W1, b1, W2, b2)
test_prediction(3, W1, b1, W2, b2)