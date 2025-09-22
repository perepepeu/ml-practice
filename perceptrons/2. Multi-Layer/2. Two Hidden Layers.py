import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Inputs (XOR)
x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])

# Expected outputs (XOR)
y = np.array([[0],
              [1],
              [1],
              [0]])

# Camadas
input_neurons = 2
hidden1_neurons = 3  # First hidden layer
hidden2_neurons = 2  # Second hidden layer
output_neurons = 1

# Random weights and biases
W1 = np.random.uniform(size=(input_neurons, hidden1_neurons))
b1 = np.random.uniform(size=(1, hidden1_neurons))
W2 = np.random.uniform(size=(hidden1_neurons, hidden2_neurons))
b2 = np.random.uniform(size=(1, hidden2_neurons))
W3 = np.random.uniform(size=(hidden2_neurons, output_neurons))
b3 = np.random.uniform(size=(1, output_neurons))

lr = 0.1
epochs = 10000

for epoch in range(epochs):
    # Forward Pass
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)
    z3 = np.dot(a2, W3) + b3
    predicted = sigmoid(z3)
    
    # Erro
    error = y - predicted

    # Backpropagation (output layer)
    d_predicted = error * sigmoid_derivative(predicted)
    error2 = d_predicted.dot(W3.T)
    d_a2 = error2 * sigmoid_derivative(a2)
    error1 = d_a2.dot(W2.T)
    d_a1 = error1 * sigmoid_derivative(a1)
    
    # Update weights and biases
    W3 += a2.T.dot(d_predicted) * lr
    b3 += np.sum(d_predicted, axis=0, keepdims=True) * lr
    W2 += a1.T.dot(d_a2) * lr
    b2 += np.sum(d_a2, axis=0, keepdims=True) * lr
    W1 += x.T.dot(d_a1) * lr
    b1 += np.sum(d_a1, axis=0, keepdims=True) * lr
    
    if epoch % 1000 == 0:
        print(f"Epoch {epoch}: Mean error: {np.mean(np.abs(error))}")

print("Final predicted output:")
print(predicted)