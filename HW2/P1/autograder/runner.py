import os
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import multiprocessing as mtp
import traceback
import sys
from helpers import *
from test import *

sys.path.append('mytorch')
from flatten import *

sys.path.append('mytorch/nn')
from pool import *
from resampling import *
from ConvTranspose import *
from Conv2d import *
from Conv1d import *
from linear import *
from activation import *
from loss import *


sys.path.append('models')
import mlp_scan as cnn_solution
import cnn

print('#################### Autograder Version v2 #########################')


np.random.seed(2020)

##########################################################################
################################   Section 4 - UpDownSampling    #########
##########################################################################


def test_upsampling_1d_correctness():

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    seeds = [11333, 11235, 11785]

    forward_res_list = []
    backward_res_list = []
    for __ in seeds:
        np.random.seed(__)
        rint = np.random.randint
        in_c, out_c = rint(5, 15), rint(5, 15)
        batch, width = rint(1, 4), rint(20, 300)
        kernel, upsampling_factor = rint(1, 10), rint(1, 10)
        x = np.random.randn(batch, in_c, width)

        upsample_1d = Upsample1d(upsampling_factor)
        forward_res = upsample_1d.forward(x)
        backward_res = upsample_1d.backward(forward_res)

        forward_res_list.append(forward_res)
        backward_res_list.append(backward_res)

    ep_fres_arr = np.empty(3, object)
    ep_bres_arr = np.empty(3, object)
    ep_fres_arr[:] = forward_res_list
    ep_bres_arr[:] = backward_res_list

    expected_res = np.load(
        'autograder/ref_result/upsample_1d_res.npz',
        allow_pickle=True)
    
    print("Expected Res:", expected_res['forward_res_list'][0].shape)
    print("Forward Res:", ep_fres_arr[0].shape)

    try:
        for i in range(3):
            assert(expected_res['forward_res_list']
                   [i].shape == ep_fres_arr[i].shape)
            assert(expected_res['backward_res_list']
                   [i].shape == ep_bres_arr[i].shape)
            assert(
                np.allclose(
                    expected_res['forward_res_list'][i],
                    ep_fres_arr[i]))
            assert(
                np.allclose(
                    expected_res['backward_res_list'][i],
                    ep_bres_arr[i]))
            print(f"Passed upsampling_1d Test: {i+1} / 3")

        print("Upsampling_1d:" + "PASS")
        print('-' * 20)
        return True
    except Exception as e:
        print("Upsampling_1d:" + "FAIL")
        print('-' * 20)
        traceback.print_exc()
        return False


def test_downsampling_1d_correctness():

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    seeds = [11333, 11235, 11785]

    forward_res_list = []
    backward_res_list = []
    for __ in seeds:
        np.random.seed(__)
        rint = np.random.randint
        in_c, out_c = rint(5, 15), rint(5, 15)
        batch, width = rint(1, 4), rint(20, 300)
        kernel, downsampling_factor = rint(1, 10), rint(1, 10)
        x = np.random.randn(batch, in_c, width)

        downsample_1d = Downsample1d(downsampling_factor)
        forward_res = downsample_1d.forward(x)
        backward_res = downsample_1d.backward(forward_res)

        forward_res_list.append(forward_res)
        backward_res_list.append(backward_res)

    ep_fres_arr = np.empty(3, object)
    ep_bres_arr = np.empty(3, object)
    ep_fres_arr[:] = forward_res_list
    ep_bres_arr[:] = backward_res_list

    expected_res = np.load(
        'autograder/ref_result/downsample_1d_res.npz',
        allow_pickle=True)

    try:
        for i in range(3):
            assert(expected_res['forward_res_list']
                   [i].shape == ep_fres_arr[i].shape)
            assert(expected_res['backward_res_list']
                   [i].shape == ep_bres_arr[i].shape)
            assert(
                np.allclose(
                    expected_res['forward_res_list'][i],
                    ep_fres_arr[i]))
            assert(
                np.allclose(
                    expected_res['backward_res_list'][i],
                    ep_bres_arr[i]))
            print(f"Passed downsampling_1d Test: {i+1} / 3")
        print("Downsampling_1d:" + "PASS")
        print('-' * 20)
        return True
    except Exception as e:
        print("Downsampling_1d:" + "FAIL")
        print('-' * 20)
        traceback.print_exc()
        return False


def test_upsampling_2d_correctness():

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    seeds = [11333, 11235, 11785]

    forward_res_list = []
    backward_res_list = []
    for __ in seeds:

        np.random.seed(__)
        rint = np.random.randint
        in_c = np.random.randint(5, 15)
        out_c = np.random.randint(5, 15)
        kernel = np.random.randint(3, 7)

        width = np.random.randint(60, 80)
        batch = np.random.randint(1, 4)
        upsampling_factor = rint(1, 10)

        x = np.random.randn(batch, in_c, width, width)

        upsample_2d = Upsample2d(upsampling_factor)
        forward_res = upsample_2d.forward(x)
        backward_res = upsample_2d.backward(forward_res)

        forward_res_list.append(forward_res)
        backward_res_list.append(backward_res)

    ep_fres_arr = np.empty(3, object)
    ep_bres_arr = np.empty(3, object)
    ep_fres_arr[:] = forward_res_list
    ep_bres_arr[:] = backward_res_list

    expected_res = np.load(
        'autograder/ref_result/upsample_2d_res.npz',
        allow_pickle=True)

    try:
        for i in range(3):
            #print("i",i)
            assert(expected_res['forward_res_list']
                   [i].shape == ep_fres_arr[i].shape)
            assert(expected_res['backward_res_list']
                   [i].shape == ep_bres_arr[i].shape)
            assert(
                np.allclose(
                    expected_res['forward_res_list'][i],
                    ep_fres_arr[i]))
            assert(
                np.allclose(
                    expected_res['backward_res_list'][i],
                    ep_bres_arr[i]))
            print(f"Passed upsampling_2d Test: {i+1} / 3")
        print("Upsampling_2d:" + "PASS")
        print('-' * 20)
        return True
    except Exception as e:
        print("Upsampling_2d:" + "FAIL")
        print('-' * 20)
        traceback.print_exc()
        return False


def test_downsampling_2d_correctness():

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    seeds = [11593, 11785, 34567]

    forward_res_list = []
    backward_res_list = []
    for __ in seeds:

        np.random.seed(__)
        rint = np.random.randint
        in_c = np.random.randint(5, 15)
        out_c = np.random.randint(5, 15)
        kernel = np.random.randint(3, 7)
        downsampling_factor = rint(1, 10)
        width = np.random.randint(60, 80)
        batch = np.random.randint(1, 4)

        x = np.random.randn(batch, in_c, width, width)

        downsample_2d = Downsample2d(downsampling_factor)
        forward_res = downsample_2d.forward(x)
        backward_res = downsample_2d.backward(forward_res)

        forward_res_list.append(forward_res)
        backward_res_list.append(backward_res)

    ep_fres_arr = np.empty(3, object)
    ep_bres_arr = np.empty(3, object)
    ep_fres_arr[:] = forward_res_list
    ep_bres_arr[:] = backward_res_list

    expected_res = np.load(
        'autograder/ref_result/downsample_2d_res.npz',
        allow_pickle=True)

    try:
        for i in range(3):
            assert(expected_res['forward_res_list']
                   [i].shape == ep_fres_arr[i].shape)
            assert(expected_res['backward_res_list']
                   [i].shape == ep_bres_arr[i].shape)
            assert(
                np.allclose(
                    expected_res['forward_res_list'][i],
                    ep_fres_arr[i]))
            assert(
                np.allclose(
                    expected_res['backward_res_list'][i],
                    ep_bres_arr[i]))
            print(f"Passed downsampling_2d Test: {i+1} / 3")

        print("Downsampling_2d:" + "PASS")
        print('-' * 20)
        return True
    except Exception as e:
        print("Downsampling_2d:" + "FAIL")
        print('-' * 20)
        traceback.print_exc()
        return False

##########################################################################
####################################   Section 5 - Convolution    ########
##########################################################################

##########################################################################
#########################   Section 5.1.1 - Conv1d_stride1  ##############
##########################################################################


def test_cnn_correctness_conv1d_stride1_once(idx):

    scores_dict = [0, 0, 0, 0]

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    rint = np.random.randint
    norm = np.linalg.norm
    in_c, out_c = rint(5, 15), rint(5, 15)
    kernel, stride = rint(1, 10), 1
    batch, width = rint(1, 4), rint(20, 300)

    def info():
        print('\nTesting model:')
        print('    in_channels: {}, out_channels: {},'.format(in_c, out_c))
        print('    kernel size: {}, stride: {},'.format(kernel, stride))
        print('    batch size: {}, input size: {}.'.format(batch, width))

    ##########################################################################
    ##########    Initialize the CNN layer and copy parameters to a PyTorch CN
    ##########################################################################

    def random_normal_weight_init_fn(out_channels, in_channels, kernel_size):
        return np.random.normal(
            0, 1.0, (out_channels, in_channels, kernel_size))

    if idx == 0:
        print("---Conv1d_stride1 Forward (Tests)---")
    try:
        net = Conv1d_stride1(
            in_c,
            out_c,
            kernel,
            random_normal_weight_init_fn,
            np.ones)
    except BaseException:
        info()
        print('Failed to pass parameters to your Conv1d function!')
        return scores_dict

    model = nn.Conv1d(in_c, out_c, kernel, stride)
    model.weight = nn.Parameter(torch.tensor(net.W))
    model.bias = nn.Parameter(torch.tensor(net.b))

    ##########################################################################
    #########################    Get the correct results from PyTorch   ######
    ##########################################################################
    x = np.random.randn(batch, in_c, width)
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = model(x1)
    torch_y = y1.detach().numpy()
    b, c, w = y1.shape
    delta = np.random.randn(b, c, w)
    y1.backward(torch.tensor(delta))

    ##########################################################################
    ##########################    Get your forward results and compare #######
    ##########################################################################
    try:
        y = net.forward(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y

    if not assertions(test_y, torch_y, 'type', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'):
        return scores_dict
    scores_dict[0] = 1

    ##########################################################################
    ##################   Get your backward results and check the tensor shape
    ##########################################################################
    dx = net.backward(delta)

    if idx == 0:
        print("---Conv1d_Stride1 Backward (Tests)---")

    assert dx.shape == x.shape
    assert net.dLdW.shape == model.weight.grad.detach().numpy().shape
    assert net.dLdb.shape == model.bias.grad.detach().numpy().shape
    ##########################################################################
    ################   Check your dLdA, dLdW and dLdb with PyTorch build-in fu
    ##########################################################################
    dx1 = x1.grad.detach().numpy()
    delta_res_norm = abs(dx - dx1).max()

    dW_res = net.dLdW - model.weight.grad.detach().numpy()
    # print(dW_res)
    dW_res_norm = abs(dW_res).max()

    db_res = net.dLdb - model.bias.grad.detach().numpy()
    db_res_norm = abs(db_res).max()

    if delta_res_norm < 1e-12:
        scores_dict[1] = 1

    if dW_res_norm < 1e-12:
        scores_dict[2] = 1

    if db_res_norm < 1e-12:
        scores_dict[3] = 1

    if min(scores_dict) != 1:
        info()
        if scores_dict[1] == 0:
            print('Fail to return correct backward values dLdA')
        if scores_dict[2] == 0:
            print('Fail to return correct backward values dLdW')
        if scores_dict[3] == 0:
            print('Fail to return correct backward values dLdb')
    return scores_dict


def test_cnn_correctness_conv1d_stride1():
    scores = []
    worker = min(mtp.cpu_count(), 4)
    p = mtp.Pool(worker)

    for __ in range(15):
        scores_dict = test_cnn_correctness_conv1d_stride1_once(__)
        scores.append(scores_dict)
        if min(scores_dict) != 1:
            return False

    a, b, c, d = np.array(scores).min(0)

    print('Section 5.1.1 - Forward')
    print('Conv1d_stride1 Forward:', 'PASS' if a == 1 else 'FAIL')

    print('Section 5.1.1 - Backward')
    print('Conv1d_stride1 dLdA:', 'PASS' if b == 1 else 'FAIL')
    print('Conv1d_stride1 dLdW:', 'PASS' if c == 1 else 'FAIL')
    print('Conv1d_stride1 dLdb:', 'PASS' if d == 1 else 'FAIL')

    print('-' * 20)
    return True

##########################################################################
####################################   Section 5.1.2 - Conv1d    #########
##########################################################################


def test_cnn_correctness_conv1d_once(idx):

    scores_dict = [0, 0, 0, 0]

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    rint = np.random.randint
    norm = np.linalg.norm
    in_c, out_c = rint(5, 15), rint(5, 15)
    kernel, stride = rint(1, 10), rint(1, 10)
    padding = rint(1, 10)
    batch, width = rint(1, 4), rint(20, 300)

    def info():
        print('\nTesting model:')
        print('    in_channels: {}, out_channels: {},'.format(in_c, out_c))
        print('    kernel size: {}, stride: {},'.format(kernel, stride))
        print('    batch size: {}, input size: {}.'.format(batch, width))

    ##########################################################################
    ##########    Initialize the CNN layer and copy parameters to a PyTorch CN
    ##########################################################################

    def random_normal_weight_init_fn(out_channels, in_channels, kernel_size):
        return np.random.normal(
            0, 1.0, (out_channels, in_channels, kernel_size))

    if idx == 0:
        print("---Conv1d Forward (Tests)---")
    try:
        net = Conv1d(
            in_c,
            out_c,
            kernel,
            stride,
            padding = padding,
            weight_init_fn = random_normal_weight_init_fn,
            bias_init_fn = np.ones
            )
    except BaseException:
        info()
        print('Failed to pass parameters to your Conv1d function!')
        return scores_dict

    model = nn.Conv1d(in_c, out_c, kernel, stride, padding=padding)
    model.weight = nn.Parameter(torch.tensor(net.conv1d_stride1.W))
    model.bias = nn.Parameter(torch.tensor(net.conv1d_stride1.b))

    ##########################################################################
    #########################    Get the correct results from PyTorch   ######
    ##########################################################################
    x = np.random.randn(batch, in_c, width)
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = model.forward(x1)
    b, c, w = y1.shape
    delta = np.random.randn(b, c, w)
    y1.backward(torch.tensor(delta))

    ##########################################################################
    ##########################    Get your forward results and compare #######
    ##########################################################################
    y = net.forward(x)
    assert y.shape == y1.shape
    if not(y.shape == y1.shape):
        print("FAILURE")

    forward_res = y - y1.detach().numpy()
    forward_res_norm = abs(forward_res).max()

    if forward_res_norm < 1e-12:
        scores_dict[0] = 1

    else:
        info()
        print('Fail to return correct forward values')
        return scores_dict

    ##########################################################################
    ##################   Get your backward results and check the tensor shape
    ##########################################################################
    dx = net.backward(delta)

    if idx == 0:
        print("---Conv1d Backward (Tests)---")
    assert dx.shape == x.shape
    assert net.conv1d_stride1.dLdW.shape == model.weight.grad.detach().numpy().shape
    assert net.conv1d_stride1.dLdb.shape == model.bias.grad.detach().numpy().shape
    ##########################################################################
    ################   Check your dx, dLdW and dLdb with PyTorch build-in func
    ##########################################################################
    dx1 = x1.grad.detach().numpy()
    delta_res_norm = abs(dx - dx1).max()

    dW_res = net.conv1d_stride1.dLdW - model.weight.grad.detach().numpy()
    dW_res_norm = abs(dW_res).max()

    db_res = net.conv1d_stride1.dLdb - model.bias.grad.detach().numpy()
    db_res_norm = abs(db_res).max()

    if delta_res_norm < 1e-12:
        scores_dict[1] = 1

    if dW_res_norm < 1e-12:
        scores_dict[2] = 1

    if db_res_norm < 1e-12:
        scores_dict[3] = 1

    if min(scores_dict) != 1:
        info()
        if scores_dict[1] == 0:
            print('Fail to return correct backward values dLdA')
        if scores_dict[2] == 0:
            print('Fail to return correct backward values dLdW')
        if scores_dict[3] == 0:
            print('Fail to return correct backward values dLdb')
    return scores_dict


def test_cnn_correctness_conv1d():
    scores = []
    worker = min(mtp.cpu_count(), 4)
    p = mtp.Pool(worker)

    for __ in range(15):
        scores_dict = test_cnn_correctness_conv1d_once(__)
        scores.append(scores_dict)
        if min(scores_dict) != 1:
            return False

    # scores = np.array(scores).min(0)
    a, b, c, d = np.array(scores).min(0)
    print('Section 5.1.2 - Forward')
    print('Conv1d Forward:', 'PASS' if a == 1 else 'FAIL')

    print('Section 5.1.2 - Backward')
    print('Conv1d dLdA:', 'PASS' if b == 1 else 'FAIL')
    print('Conv1d dLdW:', 'PASS' if c == 1 else 'FAIL')
    print('Conv1d dLdb:', 'PASS' if d == 1 else 'FAIL')

    print('-' * 20)
    return True

##########################################################################
#########################   Section 5.2.1 - Conv2d_stride1  ##############
##########################################################################


def conv2d_correctness_stride1():

    scores_dict = [0, 0, 0, 0]

    ##########################################################################
    #############################   Initialize parameters    #################
    ##########################################################################
    in_c = np.random.randint(5, 15)
    out_c = np.random.randint(5, 15)
    kernel = np.random.randint(3, 7)
    stride = 1
    width = np.random.randint(60, 80)
    batch = np.random.randint(1, 4)

    x = np.random.randn(batch, in_c, width, width)

    ##########################################################################
    #################################    Create Models   #####################
    ##########################################################################
    def random_normal_weight_init_fn(
            out_channels, in_channels, kernel_width, kernel_height):
        return np.random.normal(
            0, 1.0, (out_channels, in_channels, kernel_width, kernel_height))

    test_model = Conv2d_stride1(
        in_c,
        out_c,
        kernel,
        random_normal_weight_init_fn,
        np.zeros)

    torch_model = nn.Conv2d(in_c, out_c, kernel, stride=stride)
    torch_model.weight = nn.Parameter(torch.tensor(test_model.W))
    torch_model.bias = nn.Parameter(torch.tensor(test_model.b))

    ##########################################################################
    #########################    Get the correct results from PyTorch   ######
    ##########################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    b, c, w, h = torch_y.shape
    delta = np.random.randn(b, c, w, h)
    y1.backward(torch.tensor(delta))
    dy1 = x1.grad
    torch_dx = dy1.detach().numpy()
    torch_dW = torch_model.weight.grad.detach().numpy()
    torch_db = torch_model.bias.grad.detach().numpy()

    ##########################################################################
    ###################    Get fwd results from TestModel and compare  #######
    ##########################################################################
    try:
        y2 = test_model.forward(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2

    if not assertions(test_y, torch_y, 'type', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'):
        return scores_dict
    print('Conv2d_stride1 Forward: PASS')
    scores_dict[0] = 1

    ##########################################################################
    ###################    Get bwd results from TestModel and compare  #######
    ##########################################################################
    try:
        dy2 = test_model.backward(delta)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_dx = dy2
    test_dW = test_model.dLdW
    test_db = test_model.dLdb

    if not assertions(test_dx, torch_dx, 'type', 'dLdA'):
        return scores_dict
    if not assertions(test_dx, torch_dx, 'shape', 'dLdA'):
        return scores_dict
    if not assertions(test_dx, torch_dx, 'closeness', 'dLdA'):
        return scores_dict
    print('Conv2d_stride1 Backward dLdA: PASS')
    scores_dict[1] = 1

    if not assertions(test_dW, torch_dW, 'type', 'dLdW'):
        return scores_dict
    if not assertions(test_dW, torch_dW, 'shape', 'dLdW'):
        return scores_dict
    if not assertions(test_dW, torch_dW, 'closeness', 'dLdW'):
        return scores_dict
    print('Conv2d_stride1 Backward dLdW: PASS')
    scores_dict[2] = 1

    if not assertions(test_db, torch_db, 'type', 'dLdb'):
        return scores_dict
    if not assertions(test_db, torch_db, 'shape', 'dLdb'):
        return scores_dict
    if not assertions(test_db, torch_db, 'closeness', 'dLdb'):
        return scores_dict
    print('Conv2d_stride1 Backward dLdb: PASS')
    scores_dict[3] = 1

    ##########################################################################
    ##############################    Compare Results   ######################
    ##########################################################################

    return scores_dict


def test_conv2d_stride1():
    np.random.seed(11785)
    n = 2
    for i in range(n):
        a, b, c, d = conv2d_correctness_stride1()
        if a != 1:
            if __name__ == '__main__':
                print(
                    'Failed Conv2d_stride1 Forward Test: %d / %d' %
                    (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print(
                    'Passed Conv2d_stride1 Forward Test: %d / %d' %
                    (i + 1, n))
        if b != 1 or c != 1 or d != 1:
            if __name__ == '__main__':
                print(
                    'Failed Conv2d_stride1 Backward Test: %d / %d' %
                    (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print(
                    'Passed Conv2d_stride1 Backward Test: %d / %d' %
                    (i + 1, n))
    print('-' * 20)
    return True

##########################################################################
###############################   Section 5.2.2 - Conv2d  ################
##########################################################################


def conv2d_correctness():

    scores_dict = [0, 0, 0, 0]

    ##########################################################################
    #############################   Initialize parameters    #################
    ##########################################################################
    in_c = np.random.randint(5, 15)
    out_c = np.random.randint(5, 15)
    kernel = np.random.randint(3, 7)
    stride = np.random.randint(3, 5)
    padding = np.random.randint(3, 5)
    width = np.random.randint(60, 80)
    batch = np.random.randint(1, 4)
    # upsampling_factor = 1

    x = np.random.randn(batch, in_c, width, width)

    ##########################################################################
    #################################    Create Models   #####################
    ##########################################################################
    def random_normal_weight_init_fn(
            out_channels, in_channels, kernel_width, kernel_height):
        return np.random.normal(
            0, 1.0, (out_channels, in_channels, kernel_width, kernel_height))

    test_model = Conv2d(
        in_c,
        out_c,
        kernel,
        stride,
        padding = padding,
        weight_init_fn = random_normal_weight_init_fn,
        bias_init_fn = np.zeros)

    torch_model = nn.Conv2d(in_c, out_c, kernel, stride=stride, padding = padding)
    torch_model.weight = nn.Parameter(
        torch.tensor(test_model.conv2d_stride1.W))
    torch_model.bias = nn.Parameter(torch.tensor(test_model.conv2d_stride1.b))

    ##########################################################################
    #########################    Get the correct results from PyTorch   ######
    ##########################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1)
    torch_y = y1.detach().numpy()

    b, c, w, h = torch_y.shape
    delta = np.random.randn(b, c, w, h)
    y1.backward(torch.tensor(delta))
    dy1 = x1.grad
    torch_dx = dy1.detach().numpy()
    torch_dW = torch_model.weight.grad.detach().numpy()
    torch_db = torch_model.bias.grad.detach().numpy()

    ##########################################################################
    ###################    Get fwd results from TestModel and compare  #######
    ##########################################################################
    try:
        y2 = test_model.forward(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2

    if not assertions(test_y, torch_y, 'type', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'):
        return scores_dict
    print('Conv2d Forward : PASS')
    scores_dict[0] = 1

    ##########################################################################
    ###################    Get bwd results from TestModel and compare  #######
    ##########################################################################
    try:
        dy2 = test_model.backward(delta)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_dx = dy2
    test_dW = test_model.conv2d_stride1.dLdW
    test_db = test_model.conv2d_stride1.dLdb

    if not assertions(test_dx, torch_dx, 'type', 'dLdA'):
        return scores_dict
    if not assertions(test_dx, torch_dx, 'shape', 'dLdA'):
        return scores_dict
    if not assertions(test_dx, torch_dx, 'closeness', 'dLdA'):
        return scores_dict
    print('Conv2d Backward dLdA: PASS')
    scores_dict[1] = 1

    if not assertions(test_dW, torch_dW, 'type', 'dLdW'):
        return scores_dict
    if not assertions(test_dW, torch_dW, 'shape', 'dLdW'):
        return scores_dict
    if not assertions(test_dW, torch_dW, 'closeness', 'dLdW'):
        return scores_dict
    print('Conv2d Backward dLdW: PASS')
    scores_dict[2] = 1

    if not assertions(test_db, torch_db, 'type', 'dLdb'):
        return scores_dict
    if not assertions(test_db, torch_db, 'shape', 'dLdb'):
        return scores_dict
    if not assertions(test_db, torch_db, 'closeness', 'dLdb'):
        return scores_dict
    print('Conv2d Backward dLdb: PASS')
    scores_dict[3] = 1

    ##########################################################################
    ##############################    Compare Results   ######################
    ##########################################################################

    return scores_dict


def test_conv2d():
    np.random.seed(11785)
    n = 2
    for i in range(n):
        a, b, c, d = conv2d_correctness()
        if a != 1:
            if __name__ == '__main__':
                print('Failed Conv2d Forward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed Conv2d Forward Test: %d / %d' % (i + 1, n))
        if b != 1 or c != 1 or d != 1:
            if __name__ == '__main__':
                print('Failed Conv2d Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed Conv2d Backward Test: %d / %d' % (i + 1, n))
    print('-' * 20)
    return True

##########################################################################
###################   Section 5.3 - Transpose Convolution    #############
##########################################################################


def weight_init_fn_1d(out_channels, in_channels, kernel_width):
    np.random.seed(11785)
    return np.random.normal(0, 1.0, (out_channels, in_channels, kernel_width))


def weight_init_fn_2d(out_channels, in_channels, kernel_width1, kernel_width2):
    np.random.seed(11785)
    return np.random.normal(
        0, 1.0, (out_channels, in_channels, kernel_width1, kernel_width2))


def zeros_bias_init(d):
    return np.zeros(d)

##########################################################################
###################   Section 5.3.1 - Transpose Convolution 1d   #########
##########################################################################


def test_convTranspose_1d_correctness():

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    seeds = [11485, 11685, 11785]

    conv_forward_res_list = []
    conv_backward_res_list = []
    for __ in seeds:
        np.random.seed(__)
        rint = np.random.randint
        in_c, out_c = rint(5, 15), rint(5, 15)
        batch, width = rint(1, 4), rint(20, 300)
        kernel, upsampling_factor = rint(1, 10), rint(1, 10)
        weight_init_fn = weight_init_fn_1d
        bias_init_fn = zeros_bias_init

        x = np.random.randn(batch, in_c, width)

        conv_transpose_1d = ConvTranspose1d(in_c, out_c, kernel, upsampling_factor,
                                            weight_init_fn=weight_init_fn, bias_init_fn=bias_init_fn)

        conv_forward_res = conv_transpose_1d.forward(x)

        b, c, w = conv_forward_res.shape
        delta = np.random.randn(b, c, w)
        conv_backward_res = conv_transpose_1d.backward(delta)

        conv_forward_res_list.append(conv_forward_res)
        conv_backward_res_list.append(conv_backward_res)

    ep_fres_arr = np.empty(3, object)
    ep_bres_arr = np.empty(3, object)
    ep_fres_arr[:] = conv_forward_res_list
    ep_bres_arr[:] = conv_backward_res_list

    expected_res = np.load(
        'autograder/ref_result/convTranspose_1d_res.npz',
        allow_pickle=True)

    try:
        for i in range(3):
            assert(expected_res['conv_forward_res_list']
                   [i].shape == ep_fres_arr[i].shape)
            assert(expected_res['conv_backward_res_list']
                   [i].shape == ep_bres_arr[i].shape)
            assert(
                np.allclose(
                    expected_res['conv_forward_res_list'][i],
                    ep_fres_arr[i]))
            assert(
                np.allclose(
                    expected_res['conv_backward_res_list'][i],
                    ep_bres_arr[i]))
            print(f"Passed ConvTranspose1d Test: {i+1} / 3")
        print('-' * 20)
        return True
    except Exception as e:
        print("Failed ConvTranspose1d Test")
        print('-' * 20)
        traceback.print_exc()
        return False

##########################################################################
###################   Section 5.3.2 - Transpose Convolution 1d   #########
##########################################################################


def test_convTranspose_2d_correctness():

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    seeds = [11485, 11685, 11785]

    conv_forward_res_list = []
    conv_backward_res_list = []
    for __ in seeds:
        np.random.seed(__)
        rint = np.random.randint
        in_c, out_c = rint(3, 5), rint(3, 5)
        batch, width = rint(1, 4), rint(20, 100)
        kernel, upsampling_factor = rint(1, 3), rint(1, 3)
        weight_init_fn = weight_init_fn_2d
        bias_init_fn = zeros_bias_init

        x = np.random.randn(batch, in_c, width, width)

        conv_transpose_2d = ConvTranspose2d(in_c, out_c, kernel, upsampling_factor,
                                            weight_init_fn=weight_init_fn_2d, bias_init_fn=bias_init_fn)

        conv_forward_res = conv_transpose_2d.forward(x)

        b, c, w, h = conv_forward_res.shape
        delta = np.random.randn(b, c, w, h)
        conv_backward_res = conv_transpose_2d.backward(delta)

        conv_forward_res_list.append(conv_forward_res)
        conv_backward_res_list.append(conv_backward_res)

    ep_fres_arr = np.empty(3, object)
    ep_bres_arr = np.empty(3, object)
    ep_fres_arr[:] = conv_forward_res_list
    ep_bres_arr[:] = conv_backward_res_list

    expected_res = np.load(
        'autograder/ref_result/convTranspose_2d_res.npz',
        allow_pickle=True)

    try:
        for i in range(3):
            assert(expected_res['conv_forward_res_list']
                   [i].shape == ep_fres_arr[i].shape)
            assert(expected_res['conv_backward_res_list']
                   [i].shape == ep_bres_arr[i].shape)
            assert(
                np.allclose(
                    expected_res['conv_forward_res_list'][i],
                    ep_fres_arr[i]))
            assert(
                np.allclose(
                    expected_res['conv_backward_res_list'][i],
                    ep_bres_arr[i]))
            print(f"Passed ConvTranspose2d Test: {i+1} / 3")
        print('-' * 20)
        return True
    except Exception as e:
        print("Failed ConvTranspose2d Test")
        print('-' * 20)
        traceback.print_exc()
        return False

##########################################################################
##############################   Section 5.5 - Pooling   #################
##########################################################################

##########################################################################
###############################   Section 5.5.1 - MaxPool2d_stride1 ######
##########################################################################


def test_MaxPool2d_stride1_correctness():

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    seeds = [11485, 11685, 11785]

    maxpool_forward_res_list = []
    maxpool_backward_res_list = []
    for __ in seeds:
        np.random.seed(__)

        #######################################################################
        #############################   Initialize parameters    ##############
        #######################################################################
        kernel = np.random.randint(3, 7)
        stride = 1
        width = np.random.randint(50, 100)
        in_c = np.random.randint(5, 15)
        batch = np.random.randint(1, 4)

        x = np.random.randn(batch, in_c, width, width)

        maxpool2d_stride1 = MaxPool2d_stride1(kernel)

        maxpool_forward_res = maxpool2d_stride1.forward(x)

        maxpool_backward_res = maxpool2d_stride1.backward(maxpool_forward_res)

        maxpool_forward_res_list.append(maxpool_forward_res)
        maxpool_backward_res_list.append(maxpool_backward_res)

    ep_fres_arr = np.empty(3, object)
    ep_bres_arr = np.empty(3, object)
    ep_fres_arr[:] = maxpool_forward_res_list
    ep_bres_arr[:] = maxpool_backward_res_list

    expected_res = np.load(
        'autograder/ref_result/maxpool2d_stride1_res.npz',
        allow_pickle=True)

    try:
        for i in range(3):
            assert(expected_res['maxpool_forward_res_list']
                   [i].shape == ep_fres_arr[i].shape)
            assert(expected_res['maxpool_backward_res_list']
                   [i].shape == ep_bres_arr[i].shape)
            assert(
                np.allclose(
                    expected_res['maxpool_forward_res_list'][i],
                    ep_fres_arr[i]))
            assert(
                np.allclose(
                    expected_res['maxpool_backward_res_list'][i],
                    ep_bres_arr[i]))
            print(f"Passed Maxpool2d_stride1 Test: {i+1} / 3")
        print('-' * 20)
        return True
    except Exception as e:
        print("Failed MaxPool2d_stride1 Test")
        print('-' * 20)
        traceback.print_exc()
        return False

##########################################################################
###############################   Section 5.5.2 - MaxPool2d  #############
##########################################################################


def test_MaxPool2d_correctness():

    ##########################################################################
    #############################   Initialize hyperparameters    ############
    ##########################################################################
    seeds = [11485, 11685, 11785]

    maxpool_forward_res_list = []
    maxpool_backward_res_list = []
    for __ in seeds:
        np.random.seed(__)

        #######################################################################
        #############################   Initialize parameters    ##############
        #######################################################################
        kernel = np.random.randint(3, 7)
        stride = np.random.randint(3, 5)
        width = np.random.randint(50, 100)
        in_c = np.random.randint(5, 15)
        batch = np.random.randint(1, 4)

        x = np.random.randn(batch, in_c, width, width)

        maxpool2d = MaxPool2d(kernel, stride)

        maxpool_forward_res = maxpool2d.forward(x)

        maxpool_backward_res = maxpool2d.backward(maxpool_forward_res)

        maxpool_forward_res_list.append(maxpool_forward_res)
        maxpool_backward_res_list.append(maxpool_backward_res)

    ep_fres_arr = np.empty(3, object)
    ep_bres_arr = np.empty(3, object)
    ep_fres_arr[:] = maxpool_forward_res_list
    ep_bres_arr[:] = maxpool_backward_res_list

    expected_res = np.load(
        'autograder/ref_result/maxpool2d_res.npz',
        allow_pickle=True)

    try:
        for i in range(3):
            assert(expected_res['maxpool_forward_res_list']
                   [i].shape == ep_fres_arr[i].shape)
            assert(expected_res['maxpool_backward_res_list']
                   [i].shape == ep_bres_arr[i].shape)
            assert(
                np.allclose(
                    expected_res['maxpool_forward_res_list'][i],
                    ep_fres_arr[i]))
            assert(
                np.allclose(
                    expected_res['maxpool_backward_res_list'][i],
                    ep_bres_arr[i]))
            print(f"Passed Maxpool2d Test: {i+1} / 3")
        print('-' * 20)
        return True
    except Exception as e:
        print("Failed MaxPool2d Test")
        print('-' * 20)
        traceback.print_exc()
        return False


##########################################################################
###############################   Section 5.5.3 - MeanPool_stride1  ######
##########################################################################

def mean_pool_correctness_stride1():

    scores_dict = [0, 0]

    ##########################################################################
    #############################   Initialize parameters    #################
    ##########################################################################
    kernel = np.random.randint(3, 7)
    stride = 1
    width = np.random.randint(50, 100)
    in_c = np.random.randint(5, 15)
    batch = np.random.randint(1, 4)

    x = np.random.randn(batch, in_c, width, width)

    ##########################################################################
    #################################    Create Models   #####################
    ##########################################################################
    torch_model = nn.functional.avg_pool2d

    test_model = MeanPool2d_stride1(kernel)

    ##########################################################################
    #########################    Get the correct results from PyTorch   ######
    ##########################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1, kernel, stride)
    torch_y = y1.detach().numpy()
    y1.backward(y1)
    x1p = x1.grad
    torch_xp = x1p.detach().numpy()

    ##########################################################################
    ###################    Get fwd results from TestModel and compare  #######
    ##########################################################################
    try:
        y2 = test_model.forward(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2

    if not assertions(test_y, torch_y, 'type', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'):
        return scores_dict
    scores_dict[0] = 1

    ##########################################################################
    ###################    Get bwd results from TestModel and compare  #######
    ##########################################################################
    try:
        x2p = test_model.backward(y2)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_xp = x2p

    if not assertions(test_xp, torch_xp, 'type', 'dLdA'):
        return scores_dict
    if not assertions(test_xp, torch_xp, 'shape', 'dLdA'):
        return scores_dict
    if not assertions(test_xp, torch_xp, 'closeness', 'dLdA'):
        return scores_dict
    scores_dict[1] = 1

    return scores_dict


def test_mean_pool_stride1():
    n = 3
    np.random.seed(11785)
    for i in range(n):
        a, b = mean_pool_correctness_stride1()
        if a != 1:
            if __name__ == '__main__':
                print(
                    'Failed MeanPool2d_stride1 Forward Test: %d / %d' %
                    (i + 1, n))
            return False
        elif b != 1:
            if __name__ == '__main__':
                print(
                    'Failed MeanPool2d_stride1 Backward Test: %d / %d' %
                    (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed MeanPool2d_stride1 Test: %d / %d' % (i + 1, n))

    print('-' * 20)
    return True

##########################################################################
###############################   Section 5.5.4 - MeanPool  ##############
##########################################################################


def mean_pool_correctness():

    scores_dict = [0, 0]

    ##########################################################################
    #############################   Initialize parameters    #################
    ##########################################################################
    kernel = np.random.randint(3, 7)
    stride = np.random.randint(3, 5)
    width = np.random.randint(50, 100)
    in_c = np.random.randint(5, 15)
    batch = np.random.randint(1, 4)

    x = np.random.randn(batch, in_c, width, width)

    ##########################################################################
    #################################    Create Models   #####################
    ##########################################################################
    torch_model = nn.functional.avg_pool2d

    test_model = MeanPool2d(kernel, stride)

    ##########################################################################
    #########################    Get the correct results from PyTorch   ######
    ##########################################################################
    x1 = Variable(torch.tensor(x), requires_grad=True)
    y1 = torch_model(x1, kernel, stride)
    torch_y = y1.detach().numpy()
    y1.backward(y1)
    x1p = x1.grad
    torch_xp = x1p.detach().numpy()

    ##########################################################################
    ###################    Get fwd results from TestModel and compare  #######
    ##########################################################################
    try:
        y2 = test_model.forward(x)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_y = y2

    if not assertions(test_y, torch_y, 'type', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'shape', 'y'):
        return scores_dict
    if not assertions(test_y, torch_y, 'closeness', 'y'):
        return scores_dict
    scores_dict[0] = 1

    ##########################################################################
    ###################    Get bwd results from TestModel and compare  #######
    ##########################################################################
    try:
        x2p = test_model.backward(y2)
    except NotImplementedError:
        print("Not Implemented...")
        return scores_dict
    test_xp = x2p

    if not assertions(test_xp, torch_xp, 'type', 'dLdA'):
        return scores_dict
    if not assertions(test_xp, torch_xp, 'shape', 'dLdA'):
        return scores_dict
    if not assertions(test_xp, torch_xp, 'closeness', 'dLdA'):
        return scores_dict
    scores_dict[1] = 1

    return scores_dict


def test_mean_pool():
    n = 3
    np.random.seed(11785)
    for i in range(n):
        a, b = mean_pool_correctness()
        if a != 1:
            if __name__ == '__main__':
                print('Failed MeanPool Forward Test: %d / %d' % (i + 1, n))
            return False
        elif b != 1:
            if __name__ == '__main__':
                print('Failed MeanPool Backward Test: %d / %d' % (i + 1, n))
            return False
        else:
            if __name__ == '__main__':
                print('Passed MeanPool Test: %d / %d' % (i + 1, n))
    print('-' * 20)
    return True

##########################################################################
###############################   Section 6 - Scanning MLP    ############
##########################################################################


def test_simple_scanning_mlp():
    data = np.loadtxt(
        os.path.join(
            'autograder', 'data', 'data.asc')).T.reshape(
        1, 24, -1)
    cnn = cnn_solution.CNN_SimpleScanningMLP()
    weights = np.load(
        os.path.join(
            'autograder',
            'weights',
            'mlp_weights_part_b.npy'),
        allow_pickle=True)
    cnn.init_weights(weights)

    expected_result = np.load(
        os.path.join(
            'autograder',
            'ref_result',
            'res_b.npy'),
        allow_pickle=True)
    result = cnn.forward(data) # here

    try:
        assert(isinstance(result, type(expected_result)))
        assert(result.shape == expected_result.shape)

        assert(np.allclose(result, expected_result))

        print("Simple Scanning MLP:" + "PASS")
        print('-' * 20)
        return True
    except Exception as e:
        print("Simple Scanning MLP:" + "FAIL")
        traceback.print_exc()
        print('-' * 20)
        return False


def test_distributed_scanning_mlp():
    data = np.loadtxt(
        os.path.join(
            'autograder', 'data', 'data.asc')).T.reshape(
        1, 24, -1)
    cnn = cnn_solution.CNN_DistributedScanningMLP()
    weights = np.load(
        os.path.join(
            'autograder',
            'weights',
            'mlp_weights_part_c.npy'),
        allow_pickle=True)
    cnn.init_weights(weights)

    expected_result = np.load(
        os.path.join(
            'autograder',
            'ref_result',
            'res_c.npy'),
        allow_pickle=True)
    result = cnn.forward(data)

    try:
        assert(isinstance(result, type(expected_result)))
        assert(result.shape == expected_result.shape)
        assert(np.allclose(result, expected_result))
        print("Distributed Scanning MLP:" + "PASS")
        print('-' * 20)
        return True
    except Exception as e:
        print("Distributed Scanning MLP:" + "FAIL")
        print('-' * 20)
        traceback.print_exc()
        return False


##########################################################################
################   Section 7 - Build Your Own CNN Model    ###############
##########################################################################

# Default Weight Initialization for Conv1d and Linear

def conv1d_random_normal_weight_init(d0, d1, d2):
    return np.random.normal(0, 1.0, (d0, d1, d2))


def linear_random_normal_weight_init(d0, d1):
    return np.random.normal(0, 1.0, (d0, d1))


def zeros_bias_init(d):
    return np.zeros(d)


def get_cnn_model():
    input_width = 128
    input_channels = 24
    conv_weight_init_fn = conv1d_random_normal_weight_init
    linear_weight_init_fn = linear_random_normal_weight_init
    bias_init_fn = zeros_bias_init
    criterion = CrossEntropyLoss()
    lr = 1e-3

    num_linear_neurons = 10
    out_channels = [56, 28, 14]
    kernel_sizes = [5, 6, 2]
    strides = [1, 2, 2]
    activations = [Tanh(), ReLU(), Sigmoid()]

    model = cnn.CNN(input_width, input_channels, out_channels, kernel_sizes, strides, num_linear_neurons,
                    activations, conv_weight_init_fn, bias_init_fn, linear_weight_init_fn,
                    criterion, lr)

    model.linear_layer.W = linear_weight_init_fn(
        model.linear_layer.W.shape[0],
        model.linear_layer.W.shape[1])
    model.linear_layer.b = bias_init_fn(
        model.linear_layer.b.shape[0]).reshape(-1, 1)
    return model


class Flatten_(nn.Module):
    def __init__(self):
        super(Flatten_, self).__init__()

    def forward(self, input):
        return input.view(input.size(0), -1)


class CNN_model(nn.Module):
    def __init__(self):
        super(CNN_model, self).__init__()
        self.in_channels = 128
        self.out_size = 10

        self.conv1 = nn.Conv1d(self.in_channels, 56, 5, 1)
        self.conv2 = nn.Conv1d(56, 28, 6, 2)
        self.conv3 = nn.Conv1d(28, 14, 2, 2)

        self.flatten = Flatten_()
        self.fc = nn.Linear(14 * 30, self.out_size)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.tanh(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.sigmoid(x)

        x = self.flatten(x)
        x = self.fc(x)

        return x


def cnn_model_correctness(idx):
    scores_dict = [0, 0, 0, 0]

    TOLERANCE = 1e-8

    '''
    Write assertions to check the weight dimensions for each layer and see whether they're initialized correctly
    '''

    submit_model = get_cnn_model()
    ref_model = CNN_model()
    for i in range(3):
        getattr(ref_model, 'conv{:d}'.format(i + 1)).weight = nn.Parameter(
            torch.tensor(submit_model.convolutional_layers[i].conv1d_stride1.W))
        getattr(ref_model, 'conv{:d}'.format(i + 1)).bias = nn.Parameter(
            torch.tensor(submit_model.convolutional_layers[i].conv1d_stride1.b))
    ref_model.fc.weight = nn.Parameter(
        torch.tensor(submit_model.linear_layer.W))
    b_torch = submit_model.linear_layer.b.reshape(-1)
    ref_model.fc.bias = nn.Parameter(torch.tensor(b_torch))

    data = np.loadtxt(
        os.path.join(
            'autograder',  'data', 'data.asc')).T.reshape(
        1, 24, -1)
    labels = np.load(
        os.path.join(
            'autograder',
            'data',
            'labels.npy'),
        allow_pickle=True)

    ##########################################################################
    #########################    Get the correct results from Refrence   #####
    ##########################################################################

    # Model architecture is hardcoded
    # Width: 128 -> Int((128 - 5) / 1) + 1 = 124
    #        124 -> Int((124 - 6) / 2) + 1 = 60
    #        60  -> Int((60 - 2) / 2) + 1  = 30

    x_torch = Variable(torch.from_numpy(data), requires_grad=True)
    labels_torch = torch.tensor([0.0]).long()

    y1 = ref_model(x_torch)
    b, w = y1.shape

    criterion = nn.CrossEntropyLoss()
    loss = criterion(y1, labels_torch)
    loss.backward()

    dx_ref = x_torch.grad.detach().numpy()
    dW_ref = ref_model.conv1.weight.grad.detach().numpy()
    db_ref = ref_model.conv1.bias.grad.detach().numpy()

    ##########################################################################
    ##########################    Get your forward results and compare #######
    ##########################################################################
    y2 = submit_model.forward(data)
    assert y1.shape == y2.shape
    if not (y1.shape == y2.shape):
        print("FAILURE")

    forward_res = y2 - y1.detach().numpy()
    forward_res_norm = abs(forward_res).max()
    
    # print("y1",y1)
    # print("y2",y2)
    
    print("frn",forward_res_norm)

    if forward_res_norm < TOLERANCE:
        scores_dict[0] = 1
    else:
        print("Fail to return correct forward values")
        assert False
        return scores_dict

    ##########################################################################
    ##################   Get your backward results and check the tensor shape
    ##########################################################################
    dx = submit_model.backward(labels)
    dLdW = submit_model.convolutional_layers[0].conv1d_stride1.dLdW
    dLdb = submit_model.convolutional_layers[0].conv1d_stride1.dLdb

    assert dx.shape == data.shape
    assert dW_ref.shape == dLdW.shape
    assert db_ref.shape == dLdb.shape

    ##########################################################################
    ################   Check your dx, dLdW and dLdb with Reference ###########
    ##########################################################################
    delta_res_norm = abs(dx - dx_ref).max()

    dW_res = dLdW - dW_ref
    dW_res_norm = abs(dW_res).max()

    db_res = dLdb - db_ref
    db_res_norm = abs(db_res).max()

    if delta_res_norm < TOLERANCE:
        scores_dict[1] = 1

    if dW_res_norm < TOLERANCE:
        scores_dict[2] = 1

    if db_res_norm < TOLERANCE:
        scores_dict[3] = 1

    if min(scores_dict) != 1:
        if scores_dict[1] == 0:
            print('Fail to return correct backward values dx')
        if scores_dict[2] == 0:
            print('Fail to return correct backward values dLdW')
        if scores_dict[3] == 0:
            print('Fail to return correct backward values dLdb')
        assert False
    return scores_dict


def test_conv1d_model():
    scores = []
    worker = min(mtp.cpu_count(), 4)
    p = mtp.Pool(worker)

    scores_dict = cnn_model_correctness(0)
    scores.append(scores_dict)
    if min(scores_dict) != 1:
        return False

    # scores = np.min(scores, axis = 0)
    a, b, c, d = np.array(scores).min(0)
    # print('Section 6 - CNN Complete Model | 15 points')

    print('Conv1d Model Forward:', 'PASS' if a == 1 else 'FAIL')
    print('Conv1d Model dX:', 'PASS' if b == 1 else 'FAIL')
    print('Conv1d Model dLdW:', 'PASS' if c == 1 else 'FAIL')
    print('Conv1d Model dLdb:', 'PASS' if d == 1 else 'FAIL')
    print('-' * 20)
    return True


# TODO: add tests here with names and calling the functions.
# 'autolab' is the name on autolab I think, but you probably won't need to worry about it.
# The test functions should return True or False.
tests = [
    {
        'name': '4.1.a - Downsampling1d | 2.5 points',
        'autolab': 'Downsampling1d',
        'handler': test_downsampling_1d_correctness,
        'value': 2.5,
    },
    {
        'name': '4.1.b - Upsampling1d | 2.5 points',
        'autolab': 'Upsampling1d',
        'handler': test_upsampling_1d_correctness,
        'value': 2.5,
    },
    {
        'name': '4.2.a - Downsampling2d | 2.5 points',
        'autolab': 'Downsampling2d',
        'handler': test_downsampling_2d_correctness,
        'value': 2.5,
    },
    {
        'name': '4.2.b - Upsampling2d | 2.5 points',
        'autolab': 'Upsampling2d',
        'handler': test_upsampling_2d_correctness,
        'value': 2.5,
    },
    {
        'name': '5.1.1 - Conv1d_stride1 | 10 points',
        'autolab': 'Conv1d_stride1',
        'handler': test_cnn_correctness_conv1d_stride1,
        'value': 10,
    },
    {
        'name': '5.1.2 - Conv1d | 5 points',
        'autolab': 'Conv1d',
        'handler': test_cnn_correctness_conv1d,
        'value': 5,
    },
    {
        'name': '5.2.1 - Conv2d_stride1 | 10 points',
        'autolab': 'Conv2d_stride1',
        'handler': test_conv2d_stride1,
        'value': 10,
    },
    {
        'name': '5.2.2 - Conv2d | 5 points',
        'autolab': 'Conv2d',
        'handler': test_conv2d,
        'value': 5,
    },
    {
        'name': '5.3.1 ConvTranspose1d | 5 points',
        'autolab': 'convTranspose1d',
        'handler': test_convTranspose_1d_correctness,
        'value': 5,
    },
    {
        'name': '5.3.2 ConvTranspose2d | 5 points',
        'autolab': 'convTranspose2d',
        'handler': test_convTranspose_2d_correctness,
        'value': 5,
    },
    {
        'name': '5.5.1 - MaxPool2d_stride1 | 10 points',
        'autolab': 'MaxPool2d_stride1',
        'handler': test_MaxPool2d_stride1_correctness,
        'value': 10,
    },
    {
        'name': '5.5.2 - MaxPool2d | 5 points',
        'autolab': 'MaxPool2d',
        'handler': test_MaxPool2d_correctness,
        'value': 5,
    },
    {
        'name': '5.5.3 - MeanPool2d_stride1 | 10 points',
        'autolab': 'MeanPool2d_stride1',
        'handler': test_mean_pool_stride1,
        'value': 10,
    },

    {
        'name': '5.5.4 - MeanPool2d | 5 ponts',
        'autolab': 'MeanPool2d',
        'handler': test_mean_pool,
        'value': 5,
    },
    {
        'name': '6.1 - CNN as Simple Scanning MLP | 5 points',
        'autolab': 'CNN as Simple Scanning MLP',
        'handler': test_simple_scanning_mlp,
        'value': 5,
    },
    {
        'name': '6.2 - CNN as Distributed Scanning MLP | 5 points',
        'autolab': 'CNN as Distributed Scanning MLP',
        'handler': test_distributed_scanning_mlp,
        'value': 10,
    },
    {
        'name': '7 - Build a CNN Model | 5 points',
        'autolab': 'Build a CNN Model',
        'handler': test_conv1d_model,
        'value': 5,
    }
]


if __name__ == '__main__':
    run_tests(tests)
