inputs = [1, 2, 3]
weight = [3.1, 4.3, -2.5]
bias = 2
#output = inputs[0]*weight[0] + inputs[1]*weight[1] + inputs[2]*weight[2] + bias
#print(output)  #one neuron with 3 inputs


weight1 = [3.1, 4.3, -2.5]
bias1 = -3.2
weight2 = [3.7, 6.2, 4.5]
bias2 = 1
weight3 = [3.6, 3.5, -1.5]
bias3 = 3
output = [inputs[0]*weight1[0] + inputs[1]*weight1[1] + inputs[2]*weight1[2] + bias1,
          inputs[0]*weight2[0] + inputs[1]*weight2[1] + inputs[2]*weight2[2] + bias2, 
          inputs[0]*weight3[0] + inputs[1]*weight3[1] + inputs[2]*weight3[2] + bias3]
print(output)