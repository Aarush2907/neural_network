import numpy as np
import matplotlib.pyplot as plt

from activations import relu, sigmoid, tanh_manual

x = np.linspace(-5, 5, 200)


relu_y = relu(x)
sigmoid_y = sigmoid(x)
tanh_y = tanh_manual(x)

plt.figure(figsize=(10,6))

plt.plot(x, relu_y, label="ReLU")
plt.plot(x, sigmoid_y, label="Sigmoid")
plt.plot(x, tanh_y, label="Tanh")

plt.title("Activation Functions")
plt.xlabel("Input (x)")
plt.ylabel("Output f(x)")

plt.axhline(y=0, color="black", linewidth=0.5)
plt.axvline(x=0, color="black", linewidth=0.5)

plt.grid(True)
plt.legend()

plt.show()