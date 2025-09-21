import numpy as np

def sigmoid(x):  # Activation function
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):  # Derivative of the activation function
    return x * (1 - x)

# Input and desired output
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

y = np.array([[0],
              [1],
              [1],
              [0]])

# Weights and biases initialization
input_layer_neurons = x.shape[1]  # Number of features in input layer
hidden_layer_neurons = 2  # Number of neurons in hidden layer
output_layer_neurons = 1  # Number of neurons in output layer

# Weights and biases
W1 = np.random.uniform(size=(input_layer_neurons, hidden_layer_neurons)) 
b1 = np.random.uniform(size=(1, hidden_layer_neurons))
W2 = np.random.uniform(size=(hidden_layer_neurons, output_layer_neurons))
b2 = np.random.uniform(size=(1, output_layer_neurons))
# the use of "np.random.uniform()" is because it generates random numbers in a given range, which helps in breaking symmetry and allows the network to learn effectively.


lr = 0.1  # Learning rate

epochs = 10000  # Number of training iterations

for epoch in range(epochs):
    # Forward propagation
    hidden_layer_activation = np.dot(x, W1) + b1 # Weighted sum for hidden layer
    hidden_layer_output = sigmoid(hidden_layer_activation) # Activation for hidden layer

    output_layer_activation = np.dot(hidden_layer_output, W2) + b2 # Weighted sum for output layer
    predicted_output = sigmoid(output_layer_activation) # Activation for output layer

    # Backpropagation
    error = y - predicted_output # Error
    d_predicted_output = error * sigmoid_derivative(predicted_output) # Derivative of the loss

    error_hidden_layer = d_predicted_output.dot(W2.T) # Error for hidden layer
    d_hidden_layer = error_hidden_layer * sigmoid_derivative(hidden_layer_output) # Derivative for hidden layer

    # Update weights and biases
    W2 += hidden_layer_output.T.dot(d_predicted_output) * lr # Update weights 2
    b2 += np.sum(d_predicted_output, axis=0, keepdims=True) * lr # Update bias 2
    W1 += x.T.dot(d_hidden_layer) * lr # Update weights 1 
    b1 += np.sum(d_hidden_layer, axis=0, keepdims=True) * lr # Update bias 1

    if epoch % 1000 == 0: # Print every 1000 epochs
        print(f"Epoch {epoch}: Error: {np.mean(np.abs(error))}")

print("Final predicted output:")
print(predicted_output)
