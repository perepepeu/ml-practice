import numpy as np

def sigmoid(x): # Activation function
    return 1 / (1 + np.exp(-x))

x = 1# Input
y = 1 # Desired output

# Weights and bias
w = 0.5
b = 2

z = w * x + b # Weighted sum
a = sigmoid(z) # Activation

print(f"Output: {a}, Desired: {y}")