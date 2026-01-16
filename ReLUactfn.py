import numpy as np
#input in batches
#X = [[1, 2, 3, 2.5], [2.0, 5.0, -1.0, 2.0], [-1.5, 2.7, 3.3, -0.8]]

np.random.seed(0)

#dataset
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = spiral_data(100, 3)   

class ActivationRELU     :
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

#2 hidden layers (initialising a layer)
class Layer:
    def __init__(self, ninputs, nneurons):
          self.weights = 0.10 * np.random.randn(ninputs, nneurons)
          self.biases = np.zeros((1, nneurons)) #2d(broadcasting in numpy)
    def forward(self, inputs):
          self.output =  np.dot(inputs,self.weights) + self.biases
         

layer1= Layer(2,5)  
activation1 = ActivationRELU()

layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)
