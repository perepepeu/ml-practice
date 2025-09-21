import numpy as np

def sigmoid(x):  # Activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  # Derivative of the activation function
    return x * (1 - x)

x = 0.6  # Input
y = 1    # Desired output

# Weights and bias
w = 0.5
b = 1

lr = 0.1  # Learning rate
epochs = 10  # Number of training iterations

for epoch in range(epochs): # Training loop
    z = w * x + b  # Weighted sum
    a = sigmoid(z)  # Activation

    error = y - a  # Error

    # Update weights and bias
    dZ = error * sigmoid_derivative(a)  # Derivative of the loss

    w -= lr * dZ * x  # Update weights
    b -= lr * dZ      # Update bias

    z = w * x + b  # Weighted sum
    a = sigmoid(z)  # Activation

    print("Output after training:")
    print(f"Output: {a}, Desired: {y}")