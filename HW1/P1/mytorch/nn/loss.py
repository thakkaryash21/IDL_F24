import numpy as np


class MSELoss:

    def forward(self, A, Y):
        """
        Calculate the Mean Squared error
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: MSE Loss(scalar)

        """

        self.A = A
        self.Y = Y

        self.N = A.shape[0]
        self.C = A.shape[1]

        se = (A - Y) * (A - Y)

        col_ones = np.ones((self.C, 1))
        row_ones = np.ones((self.N, 1))

        sse = row_ones.T @ se @ col_ones

        mse = sse / (self.N * self.C)

        return mse

    def backward(self):

        dLdA = 2 * (self.A - self.Y) / (self.N * self.C)

        return dLdA


class CrossEntropyLoss:

    def forward(self, A, Y):
        """
        Calculate the Cross Entropy Loss
        :param A: Output of the model of shape (N, C)
        :param Y: Ground-truth values of shape (N, C)
        :Return: CrossEntropyLoss(scalar)

        Refer the the writeup to determine the shapes of all the variables.
        Use dtype ='f' whenever initializing with np.zeros()
        """
        self.A = A
        self.Y = Y
        self.N = A.shape[0]
        self.C = A.shape[1]

        Ones_C = np.ones((self.C, 1))
        Ones_N = np.ones((self.N, 1))

        self.softmax = np.exp(A) / np.sum(np.exp(A), axis=1).reshape(-1, 1)

        crossentropy = (-Y * np.log(self.softmax)) @ Ones_C
        sum_crossentropy = Ones_N.T @ crossentropy
        L = sum_crossentropy / self.N

        return L

    def backward(self):

        dLdA = (self.softmax - self.Y) / self.N

        return dLdA
