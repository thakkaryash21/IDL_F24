# Do not import any additional 3rd party external libraries as they will not
# be available to AutoLab and are not needed (or allowed)

import numpy as np
from resampling import *


class Conv1d_stride1:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify this method
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if weight_init_fn is None:
            self.W = np.random.normal(0, 1.0, (out_channels, in_channels, kernel_size))
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """
        self.A = A
        batch_size, in_channels, input_size = A.shape
        output_size = input_size - self.kernel_size + 1

        Z = np.zeros((batch_size, self.out_channels, output_size))

        for i in range(output_size):
            patch = A[:, :, i : i + self.kernel_size]
            Z[:, :, i] = np.tensordot(patch, self.W, axes=([1, 2], [1, 2]))

        Z += self.b.reshape(1, -1, 1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        batch_size, out_channels, output_size = dLdZ.shape
        self.dLdb = np.sum(dLdZ, axis=(0, 2))
        dLdA = np.zeros(self.A.shape)

        for i in range(self.kernel_size):
            patch = self.A[:, :, i : i + output_size]

            self.dLdW[:, :, i] = np.tensordot(dLdZ, patch, axes=([0, 2], [0, 2]))

        dLdZ_padded = np.pad(
            dLdZ, ((0, 0), (0, 0), (self.kernel_size - 1, self.kernel_size - 1))
        )
        flipped_W = np.flip(self.W, axis=2)

        for i in range(self.A.shape[2]):
            patch = dLdZ_padded[:, :, i : i + self.kernel_size]
            dLdA[:, :, i] = np.tensordot(patch, flipped_W, axes=([1, 2], [0, 2]))

        return dLdA


class Conv1d:
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=0,
        weight_init_fn=None,
        bias_init_fn=None,
    ):
        # Do not modify the variable names

        self.stride = stride
        self.pad = padding

        # Initialize Conv1d() and Downsample1d() isntance
        self.conv1d_stride1 = Conv1d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample1d = Downsample1d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_size)
        Return:
            Z (np.array): (batch_size, out_channels, output_size)
        """

        A = np.pad(A, ((0, 0), (0, 0), (self.pad, self.pad)))

        # Call Conv1d_stride1
        stride = self.conv1d_stride1.forward(A)

        # downsample
        Z = self.downsample1d.forward(stride)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_size)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_size)
        """
        # Call downsample1d backward
        dLdZ = self.downsample1d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv1d_stride1.backward(dLdZ)

        # Remove padding
        if self.pad != 0:
            dLdA = dLdA[:, :, self.pad : -self.pad]

        return dLdA
