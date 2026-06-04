import numpy as np
from numpy.typing import NDArray


class Solution:

    learning_rate = 0.01

    def train_model(
        self,
        X: NDArray[np.float64],
        Y: NDArray[np.float64],
        num_iterations: int,
        initial_weights: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        # Perform Linear regression
        # Formula: W -= a * dW * L
        # Where W is the vector of weights, a is the learning rate, dW is the derivative of the losses for each weight
        # and L is the losses for each weight.

        for _ in range(num_iterations):
            # calculate predicions from model
            predictions = X @ initial_weights

            # calculate loss (L) in prediction
            loss = Y - predictions

            # apply the derivative function (-2/N) to each weight to get the gradient (dW).
            # note N is just len(X)
            gradient = -2 * X.T / len(X)

            # perform the learning step by applying the gradient to the weights, 
            # scaled by the learning rate.
            initial_weights -= self.learning_rate * gradient @ loss

        return np.round(initial_weights, 5)
