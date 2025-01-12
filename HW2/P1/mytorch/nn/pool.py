import numpy as np
from resampling import *


class MaxPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for i in range(output_width):
            for j in range(output_height):
                section = A[:, :, i : i + self.kernel, j : j + self.kernel]
                Z[:, :, i, j] = np.max(section, axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1

        dLdA = np.zeros((batch_size, in_channels, input_width, input_height))

        for i in range(output_width):
            for j in range(output_height):
                section = self.A[:, :, i : i + self.kernel, j : j + self.kernel]
                max_section = np.max(section, axis=(2, 3), keepdims=True)
                mask = section == max_section
                dLdA[:, :, i : i + self.kernel, j : j + self.kernel] += (
                    mask * dLdZ[:, :, i, j][:, :, None, None]
                )

        return dLdA


class MeanPool2d_stride1:

    def __init__(self, kernel):
        self.kernel = kernel

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        self.A = A
        batch_size, in_channels, input_width, input_height = A.shape
        output_width = input_width - self.kernel + 1
        output_height = input_height - self.kernel + 1

        Z = np.zeros((batch_size, in_channels, output_width, output_height))

        for i in range(output_width):
            for j in range(output_height):
                section = A[:, :, i : i + self.kernel, j : j + self.kernel]
                Z[:, :, i, j] = np.mean(section, axis=(2, 3))

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdA = np.zeros(self.A.shape)
        batch_size, in_channels, output_width, output_height = dLdZ.shape
        input_width = output_width + self.kernel - 1
        input_height = output_height + self.kernel - 1

        for i in range(output_width):
            for j in range(output_height):
                dLdA[:, :, i : i + self.kernel, j : j + self.kernel] += dLdZ[
                    :, :, i, j
                ][:, :, None, None]

        dLdA /= self.kernel**2

        return dLdA


class MaxPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.maxpool2d_stride1 = MaxPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.maxpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.maxpool2d_stride1.backward(dLdZ)

        return dLdA


class MeanPool2d:

    def __init__(self, kernel, stride):
        self.kernel = kernel
        self.stride = stride

        # Create an instance of MaxPool2d_stride1
        self.meanpool2d_stride1 = MeanPool2d_stride1(kernel)
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_width, input_height)
        Return:
            Z (np.array): (batch_size, out_channels, output_width, output_height)
        """
        Z = self.meanpool2d_stride1.forward(A)
        Z = self.downsample2d.forward(Z)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_width, output_height)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_width, input_height)
        """
        dLdZ = self.downsample2d.backward(dLdZ)
        dLdA = self.meanpool2d_stride1.backward(dLdZ)

        return dLdA
