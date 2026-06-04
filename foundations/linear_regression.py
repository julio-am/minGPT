import numpy as np
from numpy.typing import NDArray

class Solution:

    def get_model_prediction(self, X: NDArray[np.float64], weights: NDArray[np.float64]) -> NDArray[np.float64]:
        # X is (n, m), weights is (m,) -> return (n,) predictions
        # Round to 5 decimal places
        prediction = np.round(np.dot(X,weights), 5)
        return prediction

    def get_error(self, model_prediction: NDArray[np.float64], ground_truth: NDArray[np.float64]) -> float:
        # Compute mean squared error between predictions and ground truth
        # Round to 5 decimal places
        scalar = 1.0 / len(model_prediction)
        errors = np.power(ground_truth - model_prediction, 2)
        return np.round(scalar * np.sum(errors), 5)


        


