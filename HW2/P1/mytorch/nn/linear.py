import numpy as np


class Linear:

    def __init__(self, in_features, out_features, debug=False):
        """
        Initialize the weights and biases with zeros
        Checkout np.zeros function.
        Read the writeup to identify the right shapes for all.
        """
        self.W = np.zeros((out_features, in_features))
        self.b = np.zeros((out_features, 1))

        self.debug = debug

    def forward(self, A):
        """
        :param A: Input to the linear layer with shape (N, C0)
        :return: Output Z of linear layer with shape (N, C1)
        Read the writeup for implementation details
        """
        self.A = np.array(A)
        self.N = self.A.shape[0]
        # self.ones will help in broadcasting
        self.Ones = np.ones((self.N, 1))

        Z = self.A @ self.W.T + self.Ones @ self.b.T

        return Z

    def backward(self, dLdZ):

        dLdA = dLdZ @ self.W
        self.dLdW = dLdZ.T @ self.A
        self.dLdb = dLdZ.T @ self.Ones

        if self.debug:

            self.dLdA = dLdA

        return dLdA
