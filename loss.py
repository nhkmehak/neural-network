import math

#categorical cross entropy loss

softmaxop = [0.7, 0.1, 0.3]
targetop = [1, 0, 0] #one hot encoding 

loss = -(math.log(softmaxop[0])*targetop[0] + math.log(softmaxop[1])*targetop[1] + math.log(softmaxop[2])*targetop[2])
print(loss)

#direct
losss = -math.log(0.7)
print(losss)
print(-math.log(0.5)) #more loss here