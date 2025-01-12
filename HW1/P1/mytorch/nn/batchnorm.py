import numpy as np


class BatchNorm1d:

    def __init__(self, num_features, alpha=0.9):

        self.alpha = alpha
        self.eps = 1e-8

        self.BW = np.ones((1, num_features))
        self.Bb = np.zeros((1, num_features))
        self.dLdBW = np.zeros((1, num_features))
        self.dLdBb = np.zeros((1, num_features))

        # Running mean and variance, updated during training, used during
        # inference
        self.running_M = np.zeros((1, num_features))
        self.running_V = np.ones((1, num_features))

    def forward(self, Z, eval=False):
        """
        The eval parameter is to indicate whether we are in the
        training phase of the problem or the inference phase.
        So see what values you need to recompute when eval is False.
        """
        self.Z = Z
        self.N = Z.shape[0]
        self.M = np.mean(Z, axis=0, keepdims=True)
        self.V = np.var(Z, axis=0, keepdims=True)

        if eval == False:
            # training mode
            self.NZ = (Z - self.M) / np.sqrt(self.V + self.eps)
            self.BZ = self.BW * self.NZ + self.Bb

            self.running_M = self.alpha * self.running_M + (1 - self.alpha) * self.M
            self.running_V = self.alpha * self.running_V + (1 - self.alpha) * self.V

            return self.BZ
        else:
            # inference mode
            NZ = (Z - self.running_M) / np.sqrt(self.running_V + self.eps)
            BZ = self.BW * NZ + self.Bb

        return BZ

    def backward(self, dLdBZ):

        self.dLdBW = np.sum(dLdBZ * self.NZ, axis=0, keepdims=True)
        self.dLdBb = np.sum(dLdBZ, axis=0, keepdims=True)

        dLdNZ = dLdBZ * self.BW

        dLdV = np.sum(
            dLdNZ * (self.Z - self.M) * -0.5 * (self.V + self.eps) ** -1.5,
            axis=0,
            keepdims=True,
        )
        dLdM = np.sum(
            dLdNZ * -1 / np.sqrt(self.V + self.eps), axis=0, keepdims=True
        ) + dLdV * np.mean(-2 * (self.Z - self.M), axis=0, keepdims=True)

        dLdZ = (
            dLdNZ / np.sqrt(self.V + self.eps)
            + dLdV * 2 * (self.Z - self.M) / self.N
            + dLdM / self.N
        )

        return dLdZ
