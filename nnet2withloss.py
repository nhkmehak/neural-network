import numpy as np

np.random.seed(0)

def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = spiral_data(100, 3) 

class Loss:
    def calculate(self, output, y):
        samplelosses = self.forward(output, y)
        dataloss = np.mean(samplelosses)
        return dataloss

class LossCategoricalCrossentropy(Loss):
    def forward(self, ypred, ytrue):
        samples = len(ypred)
        ypred_clipped = np.clip(ypred, 1e-7, 1-1e-7)  #to prevent ln0 error and instability of gradient becoming 0 when loss = 0 ln1

        if len(ytrue.shape) == 1:
            correctconfidences = ypred_clipped[range(samples), ytrue]

        elif len(ytrue.shape) == 2:
            correctconfidences = np.sum(ypred_clipped*ytrue, axis=1)

        negativelog = -np.log(correctconfidences)
        return negativelog


class ActivationRELU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

class Activationsoftmax:
    def forward(self, inputs):
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  #to avoid overflow i subtracted from max 
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities


class Layer:
    def __init__(self, ninputs, nneurons):
          self.weights = 0.10 * np.random.randn(ninputs, nneurons)
          self.biases = np.zeros((1, nneurons))
    def forward(self, inputs):
          self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer(2, 3)  
activation1 = ActivationRELU()
layer2 = Layer(3, 3) 
activation2 = Activationsoftmax()

layer1.forward(X)
print(layer1.output)
activation1.forward(layer1.output)
print(activation1.output)

layer2.forward(activation1.output)
print(layer2.output)
activation2.forward(layer2.output)
print(activation2.output[:5])



lossfunction = LossCategoricalCrossentropy()
loss = lossfunction.calculate(activation2.output, y)

print("Loss:", loss)