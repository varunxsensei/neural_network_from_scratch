'''
import math

layer_input = [4.8,1.21,2.385]

E = math.e

exp_values = []
for output in layer_input:
    exp_values.append(E**output)

softmax_denominator = sum(exp_values)
softmax_vals = []

for i in exp_values:
    softmax_vals.append(i/softmax_denominator)

print(softmax_vals)
print(sum(softmax_vals))

'''


import numpy as np

layer_input = [4.8,1.21,2.385]

exp_vals = np.exp(layer_input)

print(exp_vals)

softmax_vals = exp_vals/np.sum(exp_vals)

print(softmax_vals)
print(sum(softmax_vals))