"""
Simple neural network with one dense layer and ReLU activation.

- Uses spiral data with 3 classes.
- Layer: 2 inputs → 5 neurons.
- Prints output before and after ReLU.
"""


import numpy as np
from spiral_data import create_data

X,y = create_data(100,3)

class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

layer1 = Layer_Dense(2,5)
activation1 = ReLU()
layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)