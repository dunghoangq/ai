import numpy as np
from .nn import NeuralNetwork


def numerical_gradient(network: NeuralNetwork, X, y, parameter, eps=1e-5):
    grad = np.zeros_like(parameter)
    it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])

    while not it.finished:
        idx = it.multi_index
        original = parameter[idx]

        # f(x+h)
        parameter[idx] = original + eps
        _, loss_plus = network.forward(X, y)

        # f(x-h)
        parameter[idx] = original - eps
        _, loss_minus = network.forward(X, y)

        grad[idx] = (loss_plus - loss_minus) / (2 * eps)

        parameter[idx] = original

        it.iternext()

    return grad

def gradient_check(analytic, numeric):

    rel = (
        np.linalg.norm(analytic - numeric)
        /
        (
            np.linalg.norm(analytic)
            +
            np.linalg.norm(numeric)
            +
            1e-12
        )
    )

    print(f"Relative Error: {rel:.3e}")

    if rel < 1e-7:
        print("PASS")
    else:
        print("FAIL")