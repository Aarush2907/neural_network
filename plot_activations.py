import numpy as np

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1 / (1 + np.exp(-x)))

def tanh_manual(x: np.ndarray) -> np.ndarray:
    return ((np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x)))



x = np.array([-5, -2, 0, 3, 7])
relu_ans = relu(x)
sigmoid_ans = sigmoid(x)
tanh_ans  = tanh_manual(x)
print("relu : " , relu_ans)
print("sigmoid : " , np.round(sigmoid_ans , 5))
print("Tanh : " , np.round(tanh_ans , 5))