"""
Simple neural network with one dense layer and ReLU activation.

- Uses spiral data with 3 classes.
- Layer: 2 inputs → 5 neurons.
- Prints output before and after ReLU.
"""


import numpy as np
from spiral_data import create_data


class Layer_Dense:
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Softmax:
    def forward(self,inputs):
        exp_vals = np.exp(inputs - np.max(inputs,axis = 1,keepdims = True))
        probabilities = exp_vals/np.sum(exp_vals,axis = 1,keepdims = True)
        self.output = probabilities
        
class Loss:
    def calculate(self,output,y):
        sample_loss = self.forward(output,y)
        data_loss = np.mean(sample_loss)
        return data_loss

class Loss_CategoricalCrossEntropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred,1e-7,1-(1e-7))
        if len(y_true.shape) == 1:
            correct_conf = y_pred_clipped[range(samples),y_true]
        elif len(y_true.shape == 2):
            correct_conf = np.sum(y_pred_clipped * y_true,axis = 1)
        negative_log_likelihoods = -np.log(correct_conf)   
        return negative_log_likelihoods 



X,y = create_data(100,3)

dense1 = Layer_Dense(2,3)
activation1 = ReLU()
dense2 = Layer_Dense(3,3)
activation2 = Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_func = Loss_CategoricalCrossEntropy()

loss = loss_func.calculate(activation2.output,y)
print(loss)