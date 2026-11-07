
import numpy as np

from activations import relu, sigmoid, tanh_manual


x = np.array([-5, -2, 0, 3, 7])

relu_ans = relu(x)
sigmoid_ans = sigmoid(x)
tanh_ans = tanh_manual(x)


print("ReLU    :", relu_ans)
print("Sigmoid :", np.round(sigmoid_ans, 5))
print("Tanh    :", np.round(tanh_ans, 5))
