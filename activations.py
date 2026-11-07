import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Return the ReLU activation of the input."""
    return np.maximum(0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Return the Sigmoid activation of the input."""
    return 1 / (1 + np.exp(-x))


def tanh_manual(x: np.ndarray) -> np.ndarray:
    """Return the manually calculated Tanh activation."""
    return (
        (np.exp(x) - np.exp(-x))
        / (np.exp(x) + np.exp(-x))
    )
