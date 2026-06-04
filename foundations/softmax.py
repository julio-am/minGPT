import numpy as np
from numpy.typing import NDArray


class Solution:
    def softmax(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
        # z is a 1D NumPy array of logits
        # subtract max value to prevent integer overflow
        numerators = np.exp(z - np.max(z))
        denominator = np.sum(numerators)

        return np.round(numerators / denominator, 4)
