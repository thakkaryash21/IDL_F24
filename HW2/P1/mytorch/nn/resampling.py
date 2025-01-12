import math
import numpy as np


class Upsample1d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_width)
        """
        # implement Z

        N, C, Win = A.shape
        Wout = self.upsampling_factor * (Win - 1) + 1
        Z = np.zeros((N, C, Wout))

        for i in range(Win):
            Z[:, :, i * self.upsampling_factor] = A[:, :, i]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width)
        """
        N, C, Wout = dLdZ.shape
        Win = (Wout - 1) // self.upsampling_factor + 1
        dLdA = np.zeros((N, C, Win))

        for i in range(Win):
            dLdA[:, :, i] = dLdZ[:, :, i * self.upsampling_factor]

        return dLdA


class Downsample1d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        N, C, self.Win = A.shape
        Wout = math.ceil(self.Win / self.downsampling_factor)
        Z = np.zeros((N, C, Wout))

        for i in range(Wout):
            if i * self.downsampling_factor < self.Win:
                Z[:, :, i] = A[:, :, i * self.downsampling_factor]

        return Z

    def backward(self, dLdZ):
        N, C, Wout = dLdZ.shape
        dLdA = np.zeros((N, C, self.Win))

        for i in range(Wout):
            if i * self.downsampling_factor < self.Win:
                dLdA[:, :, i * self.downsampling_factor] = dLdZ[:, :, i]

        return dLdA


class Upsample2d:

    def __init__(self, upsampling_factor):
        self.upsampling_factor = upsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N, C, Hin, Win = A.shape
        Hout = self.upsampling_factor * (Hin - 1) + 1
        Wout = self.upsampling_factor * (Win - 1) + 1

        Z = np.zeros((N, C, Hout, Wout))

        for i in range(Hin):
            for j in range(Win):
                Z[:, :, i * self.upsampling_factor, j * self.upsampling_factor] = A[
                    :, :, i, j
                ]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C, Hout, Wout = dLdZ.shape
        Hin = (Hout - 1) // self.upsampling_factor + 1
        Win = (Wout - 1) // self.upsampling_factor + 1

        dLdA = np.zeros((N, C, Hin, Win))

        for i in range(Hin):
            for j in range(Win):
                dLdA[:, :, i, j] = dLdZ[
                    :, :, i * self.upsampling_factor, j * self.upsampling_factor
                ]

        return dLdA


class Downsample2d:

    def __init__(self, downsampling_factor):
        self.downsampling_factor = downsampling_factor

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, in_channels, output_height, output_width)
        """
        N, C, self.Hin, self.Win = A.shape
        Hout = math.ceil(self.Hin / self.downsampling_factor)
        Wout = math.ceil(self.Win / self.downsampling_factor)

        Z = np.zeros((N, C, Hout, Wout))

        for i in range(Hout):
            for j in range(Wout):
                if (
                    i * self.downsampling_factor < self.Hin
                    and j * self.downsampling_factor < self.Win
                ):
                    Z[:, :, i, j] = A[
                        :, :, i * self.downsampling_factor, j * self.downsampling_factor
                    ]

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, in_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """
        N, C, Hout, Wout = dLdZ.shape

        dLdA = np.zeros((N, C, self.Hin, self.Win))

        for i in range(Hout):
            for j in range(Wout):
                dLdA[
                    :, :, i * self.downsampling_factor, j * self.downsampling_factor
                ] = dLdZ[:, :, i, j]

        return dLdA
