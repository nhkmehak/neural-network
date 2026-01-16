import numpy as np
#input in batches
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)

#2 hidden layers (initialising a layer)
class Layer:
    def __init__(self, ninputs, nneurons):
          self.weights = 0.10 * np.random.rand(ninputs, nneurons)
          self.biases = np.zeros((1, nneurons)) #2d(broadcasting in numpy)
    def forward(self, inputs):
          self.output =  np.dot(inputs,self.weights) + self.biases
         

layer1= Layer(4,5)
layer2= Layer(5,2)

layer1.forward(X)
print(layer1.output)
layer2.forward(layer1.output)
print(layer2.output)
