import numpy as np
from math import isnan

class ActivationFunction:
    def __init__(self, compute, d_compute):
        self.compute = compute
        self.d_compute = d_compute

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))

def _d_sigmoid(x):
    sig = _sigmoid(x)
    return sig * np.array([1-e for e in sig])

def _relu(x):
    output = np.copy(x)
    for idx, item in np.ndenumerate(output):
        if item < 0: output[idx] = 0
    return output

def _d_relu(x):
    output = np.copy(x)
    for idx, item in np.ndenumerate(output):
        if item >= 0: output[idx] = 1
        else: output[idx] = 0
    return output

sigmoid = ActivationFunction(_sigmoid, _d_sigmoid)
relu = ActivationFunction(_relu, _d_relu)

class Model:

    def __init__(self, layers, activation_functions):
        self.layers = layers
        self.biases = [np.random.rand(1,n) * 0.25 - 0.125 for n in layers[1:]]
        self.weights = [np.random.rand(y,x) * 0.25 - 0.125 for x, y in zip(layers[1:],layers[:-1])]
        self.activation_functions = activation_functions

    def cost(self, test_data):
        cost = 0
        for inp, out in test_data:
            diff = self.predict(inp)-np.array(out)
            diff *= diff
            cost += np.sum(diff)
        return cost / len(test_data) / 2
    
    def predict(self, inputs, data=False):
        computed_layer = np.array(inputs,ndmin=2)
        before_activation = []
        computed_layers = [computed_layer]

        for weight, bias, activation_function in zip(self.weights, self.biases, self.activation_functions):
            computed_layer = np.dot(computed_layer, weight) + bias
            before_activation.append(computed_layer)
            computed_layer = activation_function.compute(computed_layer)
            computed_layers.append(computed_layer)

        if data:
            return (computed_layer[0], before_activation, computed_layers)
        else:
            return computed_layer[0]
    
    def fit(self, train_data, train_labels, batch_size=1, epochs=1, learn_rate=0.01):
        batch_count = int(len(train_data) / batch_size)

        for epoch in range(epochs):
            for batch_num in range(batch_count):

                start_idx = batch_num * batch_size
                end_idx = (batch_num + 1) * batch_size
                batch = zip(train_data[start_idx:end_idx], train_labels[start_idx:end_idx])

                # initialize bias & weight gradients
                bias_gradient   = [np.zeros(bias.shape)   for bias   in self.biases]
                weight_gradient = [np.zeros(weight.shape) for weight in self.weights]

                for inputs, outputs in batch:
                    delta_b, delta_w = self.gradient(inputs, outputs)
                    #print(delta_b, delta_w)
                    #input()
                    bias_gradient   = [prev + delta for prev, delta in zip(bias_gradient,   delta_b)]
                    weight_gradient = [prev + delta for prev, delta in zip(weight_gradient, delta_w)]
                
                self.biases =  [bias   - delta * learn_rate / batch_size for bias,   delta in zip(self.biases,  bias_gradient)]
                self.weights = [weight - delta * learn_rate / batch_size for weight, delta in zip(self.weights, weight_gradient)]

                print(f"batch {batch_num+1} done of {batch_count}",end="       \r")
            print(f"epoch {epoch+1} done of {epochs}            ")

    def flip(self, arr):
        return [arr[len(arr) - i - 1] for i in range(len(arr))]

    def gradient(self, inputs, outputs):

        # z values are the neuron values before the activation function is applied
        # an attempt at an explanation for what's going on:
        #  - we need to calculate the derivative of each weight & bias relative to the cost function,
        #  - so we can change them by that amount to learn the pattern.
        #  - to do this, we go in steps, and multiply every partial derivative to get the entire answer
        #  - (e.g dC/da * da/dz = dC/dz):
        #      1. calculate the derivative of the computed layer values with respect to the cost function
        #         the cost function is (a - y) ^ 2 (where a is the layer values & y is the desired output)
        #         so the derivative is 2(a - y)
        #      2. calculate the derivative of the computed layer values to the z values
        #         (just the derivative of the activation function)
        #      3.1. calculate the derivative of the biases relative to the z values
        #           bc the formula is z = w * a + b, w * a is treated as a constant so the derivates relative to the
        #           cost function are the same
        #      3.2. calculate the derivative of the weights relative to the z values
        #           z = w * a + b, so the derivative is simply the previous computed layer
        #      3.3 calculate the derivative of the previous activation layer relative to the z layers
        #          this is simply the current weights (see previous formulas)
        #      4. lastly, we treat the desired change in activation layer as the desired difference (a - y) & repeat

        prediction, z_values, computed_layers = self.predict(inputs, data=True)

        end_idx = len(self.layers) - 2
        z_delta = np.array((prediction - outputs), ndmin=2)
        z_delta *= self.activation_functions[end_idx].d_compute(z_values[end_idx])
        z_deltas = [z_delta]
        w_deltas = [np.dot(np.transpose(computed_layers[len(computed_layers) - 2]), z_delta)]

        for layer_idx in range(len(self.layers) - 2):
            layer_idx = len(self.layers) - layer_idx - 2
            layer_weights = np.transpose(self.weights[layer_idx])
            layer_zs = z_values[layer_idx - 1]
            activation_function = self.activation_functions[layer_idx]
            computed_layer = computed_layers[layer_idx - 1]

            z_delta = np.dot(z_delta, layer_weights) 
            z_delta *= activation_function.d_compute(layer_zs)
            z_deltas.append(z_delta)
            w_deltas.append(np.dot(np.transpose(computed_layer), z_delta))

        return [self.flip(z_deltas),self.flip(w_deltas)]
