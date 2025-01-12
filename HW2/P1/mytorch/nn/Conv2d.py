import numpy as np
from resampling import *


class Conv2d_stride1:
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
            self.W = np.random.normal(
                0, 1.0, (out_channels, in_channels, kernel_size, kernel_size)
            )
        else:
            self.W = weight_init_fn(out_channels, in_channels, kernel_size, kernel_size)

        if bias_init_fn is None:
            self.b = np.zeros(out_channels)
        else:
            self.b = bias_init_fn(out_channels)

        self.dLdW = np.zeros(self.W.shape)
        self.dLdb = np.zeros(self.b.shape)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """
        self.A = A
        batch_size, in_channels, input_height, input_width = A.shape

        self.output_height = input_height - self.kernel_size + 1
        self.output_width = input_width - self.kernel_size + 1

        Z = np.zeros(
            (batch_size, self.out_channels, self.output_height, self.output_width)
        )

        for i in range(self.output_height):
            for j in range(self.output_width):
                section = A[:, :, i : i + self.kernel_size, j : j + self.kernel_size]
                Z[:, :, i, j] = (
                    np.tensordot(section, self.W, axes=([1, 2, 3], [1, 2, 3])) + self.b
                )

        Z = Z + self.b.reshape(1, -1, 1, 1)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        batch_size, in_channels, input_height, input_width = self.A.shape

        self.dLdb = np.sum(dLdZ, axis=(0, 2, 3))
        dLdA = np.zeros(self.A.shape)

        for i in range(self.kernel_size):
            for j in range(self.kernel_size):
                section = self.A[
                    :, :, i : i + self.output_height, j : j + self.output_width
                ]
                self.dLdW[:, :, i, j] = np.tensordot(
                    dLdZ, section, axes=([0, 2, 3], [0, 2, 3])
                )

        dLdZ_padded = np.pad(
            dLdZ,
            (
                (0, 0),
                (0, 0),
                (self.kernel_size - 1, self.kernel_size - 1),
                (self.kernel_size - 1, self.kernel_size - 1),
            ),
        )
        W_flipped = np.flip(self.W, (3, 2))

        for i in range(self.A.shape[2]):
            for j in range(self.A.shape[3]):
                section = dLdZ_padded[
                    :, :, i : i + self.kernel_size, j : j + self.kernel_size
                ]
                dLdA[:, :, i, j] = np.tensordot(
                    section, W_flipped, axes=([1, 2, 3], [0, 2, 3])
                )

        return dLdA


class Conv2d:
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

        # Initialize Conv2d() and Downsample2d() isntance
        self.conv2d_stride1 = Conv2d_stride1(
            in_channels, out_channels, kernel_size, weight_init_fn, bias_init_fn
        )
        self.downsample2d = Downsample2d(stride)

    def forward(self, A):
        """
        Argument:
            A (np.array): (batch_size, in_channels, input_height, input_width)
        Return:
            Z (np.array): (batch_size, out_channels, output_height, output_width)
        """

        # Pad the input appropriately using np.pad() function
        padded_A = np.pad(
            A, ((0, 0), (0, 0), (self.pad, self.pad), (self.pad, self.pad))
        )

        # Call Conv2d_stride1
        stride1_out = self.conv2d_stride1.forward(padded_A)

        # downsample
        Z = self.downsample2d.forward(stride1_out)

        return Z

    def backward(self, dLdZ):
        """
        Argument:
            dLdZ (np.array): (batch_size, out_channels, output_height, output_width)
        Return:
            dLdA (np.array): (batch_size, in_channels, input_height, input_width)
        """

        # Call downsample1d backward
        dLdZ = self.downsample2d.backward(dLdZ)

        # Call Conv1d_stride1 backward
        dLdA = self.conv2d_stride1.backward(dLdZ)

        # Unpad the gradient
        if self.pad != 0:
            dLdA = dLdA[:, :, self.pad : -self.pad, self.pad : -self.pad]

        return dLdA
