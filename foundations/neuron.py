import numpy as np
from numpy.typing import NDArray


class Solution:
    def forward(self, x: NDArray[np.float64], w: NDArray[np.float64], b: float, activation: str) -> float:
        # x: 1D input array
        # w: 1D weight array (same length as x)
        # b: scalar bias
        # activation: "sigmoid" or "relu"
        # rounds to 5 decimals

        # Pre-activation: z = dot(x, w) + b
        z = np.dot(x, w) + b

        match activation:
            case "sigmoid":
                # Sigmoid: σ(z) = 1 / (1 + exp(-z))
                return round(1 / (1 + np.exp(-z)), 5)
            case "relu":
                # ReLU: max(0, z)
                return np.round(max(0.0, z), 5)
            case _:
                print("invalid activation function")
                pass



