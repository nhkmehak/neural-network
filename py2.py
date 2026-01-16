import numpy as np

inputs = [1.2, 4, 3.3]
weights = [[3.1, 4.3, -2.5], [3.7, 6.2, 4.5], [3.6, 3.5, -1.5]]
biases = [-3.2, 1, 3]

layeroutputs = []
for neuronweights, neuronbias in zip(weights, biases):
    neuronoutput = 0
    for ninput, weight in zip(inputs, neuronweights):
        neuronoutput += ninput * weight
    neuronoutput += neuronbias
    layeroutputs.append(neuronoutput)

print(layeroutputs)

output = np.dot(weights, inputs) + biases
print(output)   