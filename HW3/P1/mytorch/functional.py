import numpy as np
from .autograd_engine import *

"""
Mathematical Functionalities
    These are some IMPORTANT things to keep in mind:
    - Make sure grad of inputs are exact same shape as inputs.
    - Make sure the input and output order of each function is consistent with
        your other code.
    Optional:
    - You can account for broadcasting, but it is not required 
        in the first bonus.
"""


# NOTE: Boadcast support required for HW3
def unbroadcast(grad, shape, to_keep=0):
    while len(grad.shape) != len(shape):
        grad = grad.sum(axis=0)
    for i in range(len(shape) - to_keep):
        if grad.shape[i] != shape[i]:
            grad = grad.sum(axis=i, keepdims=True)
    return grad


def identity_backward(grad_output, a):
    """Backward for identity. Already implemented."""
    return grad_output


def add_backward(grad_output, a, b):
    """Backward for addition. Already implemented."""
    a_grad = grad_output * np.ones(a.shape)
    b_grad = grad_output * np.ones(b.shape)
    a_grad = unbroadcast(a_grad, a.shape)
    b_grad = unbroadcast(b_grad, b.shape)
    return a_grad, b_grad


def sub_backward(grad_output, a, b):
    """Backward for subtraction"""
    a_grad = grad_output * np.ones(a.shape)
    b_grad = grad_output * -np.ones(b.shape)
    a_grad = unbroadcast(a_grad, a.shape)
    b_grad = unbroadcast(b_grad, b.shape)
    return a_grad, b_grad


def matmul_backward(grad_output, a, b):
    """Backward for matrix multiplication"""
    a_grad = grad_output @ b.T
    b_grad = a.T @ grad_output
    return a_grad, b_grad


def transpose_backward(grad_output, a):
    """Backward for matrix transpose"""
    if isinstance(grad_output, np.ndarray):
        return grad_output.T
    else:
        return grad_output


def mul_backward(grad_output, a, b):
    """Backward for multiplication"""
    a_grad = grad_output * b
    b_grad = grad_output * a
    a_grad = unbroadcast(a_grad, a.shape)
    b_grad = unbroadcast(b_grad, b.shape)
    return a_grad, b_grad


def div_backward(grad_output, a, b):
    """Backward for division"""
    a_grad = grad_output / b
    b_grad = grad_output * (-a / (b**2.0))
    a_grad = unbroadcast(a_grad, a.shape)
    b_grad = unbroadcast(b_grad, b.shape)
    return a_grad, b_grad


def log_backward(grad_output, a):
    """Backward for log"""
    a_grad = grad_output / a
    return a_grad


def exp_backward(grad_output, a):
    """Backward of exponential"""
    return grad_output * np.exp(a)


def max_backward(grad_output, a):
    """Backward of max"""
    return grad_output * np.where(a > 0.0, 1.0, 0.0)


# NOTE: Required for HW3
def sum_backward(grad_output, a):
    """Backward of sum over axis=0"""
    a_grad = grad_output * np.ones_like(a)
    return a_grad


# NOTE: Tanh Change required to keep track of hidden state for BPTT for HW3
def tanh_backward(grad_output, a, state=None):
    if state is not None:
        out = grad_output * (1 - state**2), None
    else:
        out = grad_output * (1 - np.tanh(a) ** 2), None

    return out


def SoftmaxCrossEntropy_backward(grad_output, pred, ground_truth):
    """
    TODO: implement Softmax CrossEntropy Loss here. You may want to
    modify the function signature to include more inputs.
    NOTE: Since the gradient of the Softmax CrossEntropy Loss is
          is straightforward to compute, you may choose to implement
          this directly rather than rely on the backward functions of
          more primitive operations.
    """
    softmax = np.exp(pred) / np.sum(np.exp(pred), axis=1, keepdims=True)
    # Grad Output not needed: dL/dL == 1
    return (softmax - ground_truth) / pred.shape[0], None


### HW2 ###
def conv1d_stride1_backward(dLdZ, A, weight, bias):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input
    weight: Model param
    bias:   Model param

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    batch_size = dLdZ.shape[0]
    w_out = dLdZ.shape[-1]
    w_in = A.shape[-1]
    out_channels, in_channels, kernel_size = weight.shape

    # dLdb : C_out x 1
    dLdb = np.sum(dLdZ, axis=(0, 2))

    # dLdW : C_out * C_in * K
    dLdW = np.zeros_like(weight)
    for k in range(kernel_size):
        axs = ([0, 2], [0, 2])
        dLdW[:, :, k] += np.tensordot(dLdZ, A[:, :, k : k + w_out], axes=axs)

    # dLdA  : N x C_in * w_in
    # pdLdZ : N * C_out * w_in
    # W     : C_out * C_in * K

    dLdA = np.zeros((batch_size, in_channels, w_in))
    # Padding only on laxt axis
    pwidths = ((0,), (0,), (kernel_size - 1,))
    pdLdZ = np.pad(dLdZ, pad_width=pwidths, mode="constant")
    flipW = np.flip(weight, axis=2)  # Flip only on last axis

    for w in range(w_in):
        axs = ([1, 2], [0, 2])
        dLdA[:, :, w] = np.tensordot(pdLdZ[:, :, w : w + kernel_size], flipW, axes=axs)

    return dLdA, dLdW, dLdb


def conv2d_stride1_backward(dLdZ, A, weight, bias):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input
    weight: Model param
    bias:   Model param

    Returns
    -------
    dLdA, dLdW, dLdb
    """
    # NOTE: You can use code from HW2P1!
    batch_size, h_out, w_out = dLdZ.shape[0], dLdZ.shape[2], dLdZ.shape[3]
    h_in, w_in = A.shape[2], A.shape[3]
    out_channels, in_channels, kernel_size, _ = weight.shape

    # dLdb : C_out x 1
    dLdb = np.sum(dLdZ, axis=(0, 2, 3))

    # dLdW : C_out * C_in * K * K
    dLdW = np.zeros_like(weight)
    for kh in range(kernel_size):
        for kw in range(kernel_size):
            axs = ([0, 2, 3], [0, 2, 3])
            dLdW[:, :, kh, kw] += np.tensordot(
                dLdZ, A[:, :, kh : kh + h_out, kw : kw + w_out], axes=axs
            )

    # dLdA : N * C_in * h_in * w_in
    dLdA = np.zeros((batch_size, in_channels, h_in, w_in))
    # Padding only on laxt two axis
    pwidths = ((0,), (0,), (kernel_size - 1,), (kernel_size - 1,))
    pdLdZ = np.pad(dLdZ, pad_width=pwidths, mode="constant")
    flipW = np.flip(weight, axis=(2, 3))  # Flip only on last two axis

    for h in range(h_in):
        for w in range(w_in):
            axs = ([1, 2, 3], [0, 2, 3])
            dLdA[:, :, h, w] = np.tensordot(
                pdLdZ[:, :, h : h + kernel_size, w : w + kernel_size], flipW, axes=axs
            )

    return dLdA, dLdW, dLdb


def downsampling1d_backward(dLdZ, A, downsampling_factor):
    """
    Inputs
    ------
    dLdz:                   Gradient from next layer
    A:                      Input
    downsampling_factor:    NOTE: for the gradient buffer to work,
                            this has to be a np.array.

    Returns
    -------
    dLdA, dLdW, dLdb [** Change to: dLdA, None **]
    """
    # NOTE: You can use code from HW2P1!
    batch_size = dLdZ.shape[0]
    out_channels = dLdZ.shape[1]
    w_in = A.shape[-1]
    dLdA = np.zeros((batch_size, out_channels, w_in))  # Initialize Z

    for batch in range(batch_size):
        for channel in range(out_channels):
            i = 0
            for w in range(0, w_in, downsampling_factor[0]):
                dLdA[batch][channel][w] = dLdZ[batch][channel][i]
                i += 1

    return dLdA, None


def downsampling2d_backward(dLdZ, A, downsampling_factor):
    """
    Inputs
    ------
    dLdz:                   Gradient from next layer
    A:                      Input
    downsampling_factor:    NOTE: for the gradient buffer to work,
                            this has to be a np.array.

    Returns
    -------
    dLdA, dLdW, dLdb [** Change to: dLdA, None **]
    """
    # NOTE: You can use code from HW2P1!
    batch_size, out_channels, _, _ = dLdZ.shape
    _, _, h_in, w_in = A.shape
    dLdA = np.zeros_like(A)

    for batch in range(batch_size):
        for channel in range(out_channels):
            j = 0
            for h in range(0, h_in, downsampling_factor[0]):
                i = 0
                for w in range(0, w_in, downsampling_factor[0]):
                    dLdA[batch][channel][h][w] = dLdZ[batch][channel][j][i]
                    i += 1
                j += 1

    return dLdA, None


def flatten_backward(dLdZ, A):
    """
    Inputs
    ------
    dLdz:   Gradient from next layer
    A:      Input

    Returns
    -------
    dLdA
    """
    # NOTE: You can use code from HW2P1!
    dLdA = dLdZ.reshape(A.shape)
    return dLdA


# NOTE: Required for HW3
def expand_dims_backward(grad_output, a, axis):
    """
    backward for np.expand_dims(a, axis=axis)

    Inputs
    ------
    grad_output:   Gradient from next layer
    a:      Input
    axis:   (!!)Encoded as np.array([int])

    Returns
    -------
    a_grad, None
    """

    if a.shape != grad_output.shape:
        a_grad = np.squeeze(grad_output, axis=axis[0])
    else:
        a_grad = grad_output
    return a_grad, None


# NOTE: Required for HW3
def squeeze_backward(grad_output, a, axis):
    """
    backward for np.squeeze(a, axis=axis)

    Inputs
    ------
    grad_output:   Gradient from next layer
    a:      Input
    axis:   (!!)Encoded as np.array([int])

    Returns
    -------
    a_grad, None
    """

    if a.shape != grad_output.shape:
        a_grad = np.expand_dims(grad_output, axis=axis[0])
    else:
        a_grad = grad_output
    return a_grad, None


# NOTE: Required for HW3
def slice_backward(grad_output, a, indices):
    """
    backward for np.squeeze(a, axis=axis)

    Inputs
    ------
    grad_output:   Gradient from next layer
    a:         Input
    indices:   (!!)An np.index_exp[indices] obj
                encoded as np.array(indices, dtype=object)

    Returns
    -------
    a_grad, None
    """

    a_grad = np.zeros(a.shape)
    a_grad[tuple(indices)] = grad_output
    return a_grad, None


# NOTE: Optional for HW3
def ctc_loss_backward(grad, logits, input_lengths, gammas, extended_symbols):
    """

    CTC loss backard

    Calculate the gradients w.r.t the parameters and return the derivative
    w.r.t the inputs, xt and ht, to the cell.

    Input
    -----
    logits [np.array, dim=(seqlength, batch_size, len(Symbols)]:
        log probabilities (output sequence) from the RNN/GRU

    target [np.array, dim=(batch_size, padded_target_len)]:
        target sequences

    input_lengths [np.array, dim=(batch_size,)]:
        lengths of the inputs

    target_lengths [np.array, dim=(batch_size,)]:
        lengths of the target

    Returns
    -------
    dY [np.array, dim=(seq_length, batch_size, len(extended_symbols))]:
        derivative of divergence w.r.t the input symbols at each time

    """
    # No need to modify
    T, B, C = logits.shape
    dY = np.full_like(logits, 0)

    for batch_itr in range(B):
        # -------------------------------------------->
        # Computing CTC Derivative for single batch
        # Process:
        #     Truncate the target to target length
        #     Truncate the logits to input length
        #     Extend target sequence with blank
        #     Compute derivative of divergence and store them in dY
        # <---------------------------------------------

        # -------------------------------------------->
        # TODO
        # <---------------------------------------------
        logits_t = logits[: input_lengths[batch_itr], batch_itr]
        ext_symbols = extended_symbols[batch_itr]
        gamma = gammas[batch_itr]
        for r in range(gamma.shape[1]):
            dY[: input_lengths[batch_itr], batch_itr, ext_symbols[r]] -= (
                grad * gamma[:, r] / logits_t[:, ext_symbols[r]]
            )

    return dY, None, None, None
