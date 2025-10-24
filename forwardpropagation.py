import numpy as np # import Numpy library to generate 

weights = np.around(np.random.uniform(size=6), decimals=2) # initialize the weights
biases = np.around(np.random.uniform(size=3), decimals=2) # initialize the biases
print(weights)
print(biases)
x_1 = 0.3 # input 1
x_2 = 0.85 # input 2

print('x1 is {} and x2 is {}'.format(x_1, x_2))

z_11 = x_1 * weights[0] + x_2 * weights[1] + biases[0]

print('The weighted sum of the inputs at the first node in the hidden layer is {}'.format(z_11))

z_12= x_1 * weights[2] + x_2 * weights[3] + biases[1]
print ("the weighted sum of the inputts at the first node in the hidden layer is {}".format(z_12))

a_11 = 1.0 / (1.0 + np.exp(-z_11))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_11, decimals=4)))

a_12 = 1.0/(1.0+ np.exp(-z_12))
print('The activation of the first node in the hidden layer is {}'.format(np.around(a_12, decimals=4)))

z_2=a_11 * weights[4] + a_12 * weights[5] + biases[2]
print('The weighted sum of the inputs at the node in the output layer is {}'.format(np.around(z_2, decimals=4)))

a_2=1.0/(1.0+np.exp(-z_2))
print('The output of the network for x1 = 0.5 and x2 = 0.85 is {}'.format(np.around(a_2, decimals=4)))