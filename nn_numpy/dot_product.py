"""
dot_product.py

This file contains a simple function to calculate the output of a neural network layer using:
    output = input • weights + bias

It's a basic operation used in fully connected layers where we multiply the input with the weights 
and add the bias.

Example usage:
    from dot_product import compute_output
    out = compute_output(inputs, weights, bias)
"""

# A simple example of a dot product operation with a bias added.
# This simulates the output of a single neuron in a neural network layer.

inputs = [0.2,1.8,0.5,3]

weights = [-0.5,0.2,-0.91,0.76]

bias = 2

output = inputs[0]*weights[0] + inputs[1]*weights[1] + inputs[2]*weights[2] + inputs[3]*weights[3] + bias

print(output)

"""
Simulates the output of a fully connected neural network layer with 3 neurons.

Each neuron:
- Takes 4 inputs
- Has its own weights and bias
- Produces an output by performing a dot product with the inputs and adding its bias

The final output is a list with one value per neuron.
"""

weight_matrix = [[0.25, -0.8, 0.1, 0.9],
                 [-0.4, 0.33, -0.75, 0.6],
                  [0.7, -0.1, 0.05, -0.95]]

biases = [2,1.2,3]

layer_output = []
for neuron_weights,neuron_bias in zip(weight_matrix,biases):
    neuron_output = 0
    for n_input,n_weight in zip(inputs,neuron_weights):
        neuron_output += (n_input*n_weight)
    neuron_output += neuron_bias
    layer_output.append(neuron_output)

print(layer_output)

# This code performs a forward pass through a dense layer using NumPy.

import numpy as np

output_matrix = np.dot(weight_matrix,inputs) + biases

print(output_matrix)