"""
This code builds a simple 2-layer neural network using NumPy.

- First part: Manual forward pass using predefined weights and biases.
- Second part: Uses a Layer_Dense class to simulate two dense layers:
    - Layer 1: 4 inputs → 5 neurons
    - Layer 2: 5 inputs → 2 neurons

It performs a forward pass through both layers and prints the final output.
"""



import numpy as np

'''
inputs = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]

weights1 = [[0.2, 0.8, -0.5, 1.0],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]]

biases1 = [2,3,0.5]

weights2 = [[0.1, -0.14, 0.5],
    [-0.5, -0.12, -0.33],
    [-0.44, 0.73, -0.13]]

biases2 = [-1,2,-0.5]

layer1_output = np.dot(inputs,np.array(weights1).T) + biases1


layer2_output = np.dot(layer1_output,np.array(weights2).T) + biases2


print(layer2_output)

'''



X = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-0.8]]

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

layer1 = Layer_Dense(4,5)
layer2 = Layer_Dense(5,2)

layer1.forward(X)
# print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
