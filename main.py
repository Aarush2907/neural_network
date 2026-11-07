import numpy as np
import random
input_array = np.random.randint(1,1000,size=(3,))
weight_array = np.random.randint(1,10,size=(3,))
bias = random.randint(1,10)

weighted_sum = np.dot(input_array , weight_array) + bias

print(weighted_sum)