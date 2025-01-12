# -*- coding: utf-8 -*-
"""
────────────────────────────────────────────────────────────────────────────────────
HW1P1 - SOLUTIONS
────────────────────────────────────────────────────────────────────────────────────
"""

import json
from mytorch.nn import Identity, Sigmoid, Tanh, ReLU, GELU, Softmax
from mytorch.nn import MSELoss, CrossEntropyLoss
from mytorch.nn import Linear
from mytorch.nn import BatchNorm1d
from mytorch.optim import SGD
from models import MLP0, MLP1, MLP4
from hw1p1_autograder_flags import *

import torch
import numpy as np

np.set_printoptions(
    suppress=True,
    precision=4)


autograder_version = '3.0.1'
print("Autograder version: " + str(autograder_version))
SEED = 0
np.random.seed(SEED)
n_tests = 3
atol_threshold = 1e-4

"""
────────────────────────────────────────────────────────────────────────────────────
# Linear Layer
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_LINEAR = DEBUG_AND_GRADE_LINEAR_flag

if DEBUG_AND_GRADE_LINEAR:

    print("──────────────────────────────────────────")
    print("LINEAR | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    W = np.array([
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    b = np.array([
        [-1.],
        [0.],
        [1.]], dtype="f")

    linear = Linear(2, 3, debug=True)
    linear.W = W
    linear.b = b

    Z = linear.forward(A)
    print("Z =\n", Z.round(4), sep="")

    dLdZ = np.array([
        [-4., -3., -2.],
        [-1., -0., 1.],
        [2., 3., 4.],
        [5., 6., 7.]], dtype="f")

    dLdA = linear.backward(dLdZ)

    dLdA = linear.dLdA
    print("\ndLdA =\n", dLdA, sep="")

    dLdW = linear.dLdW
    print("\ndLdW =\n", dLdW, sep="")

    dLdb = linear.dLdb
    print("\ndLdb =\n", dLdb, sep="")

    print("\n──────────────────────────────────────────")
    print("LINEAR | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    Z_solution = np.array([
        [10., -3., -16.],
        [4., -1., -6.],
        [-2., 1., 4.],
        [-8., 3., 14.]], dtype="f")

    dLdA_solution = np.array([
        [4., -5.],
        [4., 4.],
        [4., 13.],
        [4., 22.]], dtype="f")

    print("\ndLdA =\n", dLdA_solution, sep="")

    dLdW_solution = np.array([
        [28., 30.],
        [24., 30.],
        [20., 30.]], dtype="f")
    
    print("\ndLdW =\n", dLdW_solution, sep="")

    dLdb_solution = np.array([
        [2.],
        [6.],
        [10.]], dtype="f")
    
    print("\ndLdb =\n", dLdb_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("LINEAR | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_linear_Z = np.allclose(Z.round(4), Z_solution, atol=atol_threshold)
    print("Test Z:   ", TEST_linear_Z)

    TEST_linear_dLdA = np.allclose(dLdA.round(4), dLdA_solution, atol=atol_threshold)
    print("Test dLdA:", TEST_linear_dLdA)

    TEST_linear_dLdW = np.allclose(dLdW.round(4), dLdW_solution, atol=atol_threshold)
    print("Test dLdW:", TEST_linear_dLdW)

    TEST_linear_dLdb = np.allclose(dLdb.round(4), dLdb_solution, atol=atol_threshold)
    print("Test dLdb:", TEST_linear_dLdb)

else:

    TEST_linear_Z = False
    TEST_linear_dLdA = False
    TEST_linear_dLdW = False
    TEST_linear_dLdb = False

"""
────────────────────────────────────────────────────────────────────────────────────
# Activations
────────────────────────────────────────────────────────────────────────────────────
"""

"""
────────────────────────────────────────────────────────────────────────────────────
## Identity
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_IDENTITY = DEBUG_AND_GRADE_IDENTITY_flag

if DEBUG_AND_GRADE_IDENTITY:

    print("──────────────────────────────────────────")
    print("IDENTITY | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [0, 1],
        [2, 3]], dtype="f")

    identity = Identity()

    A = identity.forward(Z)
    print("\nA =\n", A, sep="")

    dLdA = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")
    dLdZ = identity.backward(dLdA)
    print("\ndAdZ =\n", dLdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("IDENTITY | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    dLdZ_solution = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    print("\nA =\n", A_solution, sep="")
    print("\ndAdZ =\n", dLdZ_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("IDENTITY | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("           Pass?")

    TEST_identity_A = np.allclose(A.round(4), A_solution, atol=atol_threshold)
    print("Test A:   ", TEST_identity_A)

    TEST_identity_dLdZ = np.allclose(dLdZ.round(4), dLdZ_solution, atol=atol_threshold)
    print("Test dAdZ:", TEST_identity_dLdZ)

else:

    TEST_identity_A = False
    TEST_identity_dLdZ = False

"""
────────────────────────────────────────────────────────────────────────────────────
## Sigmoid
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_SIGMOID = DEBUG_AND_GRADE_SIGMOID_flag

if DEBUG_AND_GRADE_SIGMOID:

    print("\n──────────────────────────────────────────")
    print("SIGMOID | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [0, 1],
        [2, 3]], dtype="f")

    sigmoid = Sigmoid()

    A = sigmoid.forward(Z)
    print("\nA =\n", A, sep="")

    dLdA = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")
    dLdZ = sigmoid.backward(dLdA)
    print("\ndLdZ =\n", dLdZ, sep="")

    print("──────────────────────────────────────────")
    print("SIGMOID | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [0.018, 0.0474],
        [0.1192, 0.2689],
        [0.5, 0.7311],
        [0.8808, 0.9526]], dtype="f")

    dLdZ_solution = np.array([
        [-0.0707, -0.1355],
        [-0.21  , -0.1966],
        [ 0.    ,  0.1966],
        [ 0.21  ,  0.1355]], dtype="f")

    print("\nA =\n", A_solution, sep="")

    print("\ndAdZ =\n", dLdZ_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("SIGMOID | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_sigmoid_A = np.allclose(A.round(4), A_solution, atol=atol_threshold)
    print("Test A:   ", TEST_sigmoid_A)

    TEST_sigmoid_dLdZ = np.allclose(dLdZ.round(4), dLdZ_solution, atol=atol_threshold)
    print("Test dLdZ:", TEST_sigmoid_dLdZ)

else:

    TEST_sigmoid_A = False
    TEST_sigmoid_dLdZ = False

"""
────────────────────────────────────────────────────────────────────────────────────
## Tanh
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_TANH = DEBUG_AND_GRADE_TANH_flag

if DEBUG_AND_GRADE_TANH:

    print("\n──────────────────────────────────────────")
    print("TANH | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [0, 1],
        [2, 3]], dtype="f")

    tanh = Tanh()

    A = tanh.forward(Z)
    print("\nA =\n", A, sep="")

    dLdA = np.array([
        [1.0,   1.0,],
        [3.0,   1.0,],
        [2.0,   0.0,],
        [0.0,  -1.0,]], dtype="f")


    dLdZ = tanh.backward(dLdA)
    print("\ndLdZ =\n", dLdZ, sep="")

    print("──────────────────────────────────────────")
    print("TANH | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [-0.9993, -0.9951],
        [-0.964, -0.7616],
        [0., 0.7616],
        [0.964, 0.9951]], dtype="f")

    dLdZ_solution = np.array([
        [ 1.300e-03,  9.900e-03],
        [ 2.121e-01,  4.200e-01],
        [ 2.000e+00,  0.000e+00],
        [ 0.000e+00, -9.900e-03]], dtype="f")

    print("\nA =\n", A_solution, sep="")
    print("\ndAdZ =\n", dLdZ_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("TANH | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_tanh_A = np.allclose(A.round(4), A_solution, atol=atol_threshold)
    print("Test A:   ", TEST_tanh_A)

    TEST_tanh_dLdZ = np.allclose(dLdZ.round(4), dLdZ_solution, atol=atol_threshold)
    print("Test dLdZ:", TEST_tanh_dLdZ)

else:

    TEST_tanh_A = False
    TEST_tanh_dLdZ = False

"""
────────────────────────────────────────────────────────────────────────────────────
## ReLU
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_RELU = DEBUG_AND_GRADE_RELU_flag

if DEBUG_AND_GRADE_RELU:

    print("\n──────────────────────────────────────────")
    print("RELU | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([
        [-4, -3],
        [-2, -1],
        [0, 1],
        [2, 3]], dtype="f")

    relu = ReLU()

    A = relu.forward(Z)
    print("A =\n", A, sep="")

    dLdA = np.array([
        [1.0,   1.0,],
        [3.0,   1.0,],
        [2.0,   0.0,],
        [0.0,  -1.0,]], dtype="f")
    dLdZ = relu.backward(dLdA)
    print("\ndLdZ =\n", dLdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("RELU | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    A_solution = np.array([
        [0., 0.],
        [0., 0.],
        [0., 1.],
        [2., 3.]], dtype="f")

    dLdZ_solution = np.array([
        [0., 0.],
        [0., 0.],
        [0., 0.],
        [0., -1.]], dtype="f")

    print("\nA =\n", A_solution, "\n", sep="")
    print("\ndLdZ =\n", dLdZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("RELU | CLOSENESS TEST RESULT")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_relu_A = np.allclose(A.round(4), A_solution, atol=atol_threshold)
    print("Test A:   ", TEST_relu_A)

    TEST_relu_dLdZ = np.allclose(dLdZ.round(4), dLdZ_solution, atol=atol_threshold)
    print("Test dLdZ:", TEST_relu_dLdZ)

else:

    TEST_relu_A = False
    TEST_relu_dLdZ = False

"""
────────────────────────────────────────────────────────────────────────────────────
## GELU
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_GELU = DEBUG_AND_GRADE_GELU_flag

if DEBUG_AND_GRADE_GELU:

    print("\n──────────────────────────────────────────")
    print("GELU | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([-2.0, -0.5, 0.0, 0.5, 2.0])

    gelu = GELU()

    A = gelu.forward(Z)
    print("A =\n", A, sep="")

    dLdA = np.array([1.0, 1.0, 0, 1.0, -1.0])
    dLdZ = gelu.backward(dLdA)
    print("\ndLdZ =\n", dLdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("GELU | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")
   
    A_solution = np.array([-0.0455, -0.1543, 0.0, 0.3457, 1.9545])

    

    dLdZ_solution = np.array([-0.0852, 0.1325, 0, 0.8675, -1.0852])

    print("\nA =\n", A_solution, "\n", sep="")
    print("\ndLdZ =\n", dLdZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("GELU | CLOSENESS TEST RESULT")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_gelu_A = np.allclose(A.round(4), A_solution, atol=atol_threshold)
    print("Test A:   ", TEST_gelu_A)

    TEST_gelu_dLdZ = np.allclose(dLdZ.round(4), dLdZ_solution, atol=atol_threshold)
    print("Test dLdZ:", TEST_gelu_dLdZ)

else:

    TEST_gelu_A = False
    TEST_gelu_dLdZ = False


"""
────────────────────────────────────────────────────────────────────────────────────
## SOFTMAX
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_SOFTMAX = DEBUG_AND_GRADE_SOFTMAX_flag

if DEBUG_AND_GRADE_SOFTMAX:

    print("\n──────────────────────────────────────────")
    print("SOFTMAX | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    Z = np.array([[3.0, -4.5, 1.0, 6.5, 2.0], [-2.0, -0.5, 0.0, 0.5, 2.0]])

    softmax = Softmax()

    A = softmax.forward(Z)
    print("A =\n", A, sep="")

    dLdA = np.array([[2.0, -1.0, 3.0, -1.0, -2.0], [1.0, 1.0, 3.0, 1.0, -1.0]])
    dLdZ = softmax.backward(dLdA)
    print("\ndLdZ =\n", dLdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("SOFTMAX | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")
   
    A_solution = np.array([[0.0289, 0.0    , 0.0039, 0.9566, 0.0106],
       [0.0126, 0.0563, 0.0928, 0.1529, 0.6855]]) 

    dLdZ_solution = np.array([[ 0.084 , 0.0    ,  0.0153, -0.0877, -0.0116],
       [ 0.0149,  0.0667,  0.2955,  0.1813, -0.5584]])

    print("\nA =\n", A_solution, "\n", sep="")
    print("\ndLdZ =\n", dLdZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("SOFTMAX | CLOSENESS TEST RESULT")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_softmax_A = np.allclose(A.round(4), A_solution, atol=atol_threshold)
    print("Test A:   ", TEST_softmax_A)

    TEST_softmax_dLdZ = np.allclose(dLdZ.round(4), dLdZ_solution, atol=atol_threshold)
    print("Test dLdZ:", TEST_softmax_dLdZ)

else:

    TEST_softmax_A = False
    TEST_softmax_dLdZ = False

"""
────────────────────────────────────────────────────────────────────────────────────
# Multilayer Perceptrons
────────────────────────────────────────────────────────────────────────────────────
"""

"""
────────────────────────────────────────────────────────────────────────────────────
## MLP0
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_MLP0 = DEBUG_AND_GRADE_MLP0_flag

if DEBUG_AND_GRADE_MLP0:
    print("\n──────────────────────────────────────────")
    print("MLP0")
    print("──────────────────────────────────────────")

    for i in range(n_tests):
        TEST_mlp0_Z0 = True
        TEST_mlp0_A1 = True
        TEST_mlp0_dLdZ0 = True
        TEST_mlp0_dLdA0 = True
        TEST_mlp0_dLdW0 = True
        TEST_mlp0_dLdb0 = True

        A0 = np.random.randn(4, 2).astype("f")
        W0 = np.random.randn(3, 2).astype("f")
        b0 = np.random.randn(3, ).astype("f")
        A0_tensor = torch.tensor(A0, requires_grad=True)

        # Test forward
        # Use torch.linear to get the correct answer
        torch_linear = torch.nn.Linear(2, 3)
        torch_linear.weight.data = torch.tensor(W0)
        torch_linear.bias.data = torch.tensor(b0)
        torch_linear.requires_grad_()
        Z0_tensor = torch_linear(A0_tensor)
        Z0 = Z0_tensor.detach().numpy()
        A1_tensor = torch.relu(Z0_tensor)
        A1 = A1_tensor.detach().numpy()

        # Use your Linear layer to get the student answer
        mlp0 = MLP0(debug=True)

        if not mlp0.layers[0].W.shape == W0.shape:
            print ("Incorrect architecture")
            TEST_mlp0_Z0 = False
            TEST_mlp0_A1 = False
            break

        mlp0.layers[0].W = W0
        mlp0.layers[0].b = b0.reshape(-1, 1)
        A1_ = mlp0.forward(A0)
        Z0_ = mlp0.Z0
        

        # Compare the student answer with the correct answer
        if np.allclose(Z0, Z0_, atol=atol_threshold)==False:
            TEST_mlp0_Z0=False
            
        if np.allclose(A1, A1_, atol=atol_threshold)==False:
            TEST_mlp0_A1=False
        
        if TEST_mlp0_Z0 and TEST_mlp0_A1:
            print("Passed Forward on testcase", i + 1)
        else:
            print("Failed Forward on testcase", i + 1)
            print("A0:\n", A0, sep="")
            print("W0:\n", W0, sep="")
            print("b0:\n", b0, sep="")
            print("Your Z0:\n", Z0_, sep="")
            print("Correct Z0:\n", Z0, sep="")
            print("Your A1:\n", A1_, sep="")
            print("Correct A1:\n", A1, sep="")
            break

        
        # Test backwards
        # Use torch.linear to get the correct answer
        dLdA1 = np.random.randn(4, 3).astype("f")
        dLdA1_tensor = torch.tensor(dLdA1) 
        dA1dZ0 = torch.autograd.grad(A1_tensor, Z0_tensor, grad_outputs=torch.ones_like(A1_tensor))[0].numpy()    
        dLdZ0 = dLdA1*dA1dZ0
        Z0_tensor.backward(gradient=torch.tensor(dLdZ0))
        dLdA0 = A0_tensor.grad.data.numpy()
        dLdW0 = torch_linear.weight.grad.data.numpy()
        dLdb0 = torch_linear.bias.grad.data.numpy().reshape(-1, 1)

        mlp0.backward(dLdA1)
        dLdA0_ = mlp0.dLdA0
        dLdZ0_ = mlp0.dLdZ0
        dLdW0_ = mlp0.layers[0].dLdW
        dLdb0_ = mlp0.layers[0].dLdb

        # Compare the student answer with the correct answer
  
        if np.allclose(dLdZ0, dLdZ0_, atol=atol_threshold)==False:
            TEST_mlp0_dLdZ0=False

        if np.allclose(dLdA0, dLdA0_, atol=atol_threshold)==False:
            TEST_mlp0_dLdA0=False
            
        if np.allclose(dLdW0, dLdW0_, atol=atol_threshold)==False:
            TEST_mlp0_dLdW0=False
            
        if np.allclose(dLdb0, dLdb0_, atol=atol_threshold)==False:
            TEST_mlp0_dLdb0=False

        
        if TEST_mlp0_A1 and TEST_mlp0_dLdA0 and TEST_mlp0_dLdW0 and TEST_mlp0_dLdb0 and TEST_mlp0_dLdZ0:
            print("Passed Backward on testcase", i + 1)
        else:
            print("Failed Backward on testcase", i + 1)
            print("A0:\n", A0, sep="")
            print("W0:\n", W0, sep="")
            print("b0:\n", b0, sep="")
            break

else:

    TEST_mlp0_Z0 = False
    TEST_mlp0_A1 = False
    TEST_mlp0_dA1dZ0 = False
    TEST_mlp0_dLdZ0 = False
    TEST_mlp0_dLdA0 = False
    TEST_mlp0_dLdW0 = False
    TEST_mlp0_dLdb0 = False

"""## MPL1"""

DEBUG_AND_GRADE_MLP1 = DEBUG_AND_GRADE_MLP1_flag

if DEBUG_AND_GRADE_MLP1:

    print("──────────────────────────────────────────")
    print("MLP1")
    print("──────────────────────────────────────────")

    for i in range(n_tests):
        # Test forward
        TEST_mlp1_Z0 = True
        TEST_mlp1_A1 = True
        TEST_mlp1_Z1 = True
        TEST_mlp1_A2 = True

        A0 = np.random.randn(4, 2).astype("f")
        W0 = np.random.randn(3, 2).astype("f")
        b0 = np.random.randn(3, ).astype("f")
        W1 = np.random.randn(2, 3).astype("f")
        b1 = np.random.randn(2, ).astype("f")
        A0_tensor = torch.tensor(A0, requires_grad=True)

        # Use torch.linear to get the correct answer
        torch_linear0 = torch.nn.Linear(2, 3)
        torch_linear0.weight.data = torch.tensor(W0)
        torch_linear0.bias.data = torch.tensor(b0)
        torch_linear0.requires_grad_()
        torch_linear1 = torch.nn.Linear(3, 2)
        torch_linear1.weight.data = torch.tensor(W1)
        torch_linear1.bias.data = torch.tensor(b1)
        torch_linear1.requires_grad_()
        Z0_tensor = torch_linear0(A0_tensor)
        Z0 = Z0_tensor.detach().numpy()
        A1_tensor = torch.relu(Z0_tensor)
        A1 = A1_tensor.detach().numpy()
        A1_tensor_copy = torch.tensor(A1, requires_grad=True)
        Z1_tensor = torch_linear1(A1_tensor_copy)
        Z1 = Z1_tensor.detach().numpy()
        A2_tensor = torch.relu(Z1_tensor)
        A2 = A2_tensor.detach().numpy()

        # Use your Linear layer to get the student answer
        mlp1 = MLP1(debug=True)

        if not (mlp1.layers[0].W.shape == W0.shape and mlp1.layers[2].W.shape == W1.shape):
            print ("Incorrect architecture")
            TEST_mlp1_Z0 = False
            TEST_mlp1_A1 = False
            TEST_mlp1_Z1 = False
            TEST_mlp1_A2 = False
            break

        mlp1.layers[0].W = W0
        mlp1.layers[0].b = b0.reshape(-1, 1)
        mlp1.layers[2].W = W1
        mlp1.layers[2].b = b1.reshape(-1, 1)
        A2_ = mlp1.forward(A0)
        Z0_ = mlp1.Z0
        Z1_ = mlp1.Z1
        A1_ = mlp1.A1

        # Compare the student answer with the correct answer
        if np.allclose(Z0, Z0_, atol=atol_threshold)==False:
            TEST_mlp1_Z0=False
        
        if np.allclose(A1, A1_, atol=atol_threshold)==False:
            TEST_mlp1_A1=False

        if np.allclose(Z1, Z1_, atol=atol_threshold)==False:
            TEST_mlp1_Z1=False

        if np.allclose(A2, A2_, atol=atol_threshold)==False:
            TEST_mlp1_A2=False

        if TEST_mlp1_Z0 and TEST_mlp1_A1 and TEST_mlp1_Z1 and TEST_mlp1_A2:
            print("Passed Forward on testcase", i + 1)

        else:
            print("Failed Forward on testcase", i + 1)
            print("A0:\n", A0, sep="")
            print("W0:\n", W0, sep="")
            print("b0:\n", b0, sep="")
            print("W1:\n", W1, sep="")
            print("b1:\n", b1, sep="")
            print("Your Z0:\n", Z0_, sep="")
            print("Correct Z0:\n", Z0, sep="")
            print("Your A1:\n", A1_, sep="")
            print("Correct A1:\n", A1, sep="")
            print("Your Z1:\n", Z1_, sep="")
            print("Correct Z1:\n", Z1, sep="")
            print("Your A2:\n", A2_, sep="")
            print("Correct A2:\n", A2, sep="")
            break


        # Test backwards
        TEST_mlp1_dA2dZ1 = True
        TEST_mlp1_dLdZ1 = True
        TEST_mlp1_dLdA1 = True
        TEST_mlp1_dLdZ0 = True
        TEST_mlp1_dLdA0 = True
        TEST_mlp1_dLdW0 = True
        TEST_mlp1_dLdb0 = True
        TEST_mlp1_dLdW1 = True
        TEST_mlp1_dLdb1 = True

        # Use torch.linear to get the correct answer
        dLdA2 = np.random.randn(4, 2).astype("f")
        dLdA2_tensor = torch.tensor(dLdA2)
        dA2dZ1 = torch.autograd.grad(A2_tensor, Z1_tensor, grad_outputs=torch.ones_like(A2_tensor))[0].numpy()
        dLdZ1 = dLdA2*dA2dZ1
        Z1_tensor.backward(gradient=torch.tensor(dLdZ1), retain_graph=True)
        dLdA1 = A1_tensor_copy.grad.data.numpy()
        dLdW1 = torch_linear1.weight.grad.data.numpy()
        dLdb1 = torch_linear1.bias.grad.data.numpy().reshape(-1, 1)
        dA1dZ0 = torch.autograd.grad(A1_tensor, Z0_tensor, grad_outputs=torch.ones_like(A1_tensor))[0].numpy()
        dLdZ0 = dLdA1*dA1dZ0
        Z0_tensor.backward(gradient=torch.tensor(dLdZ0), retain_graph=True)
        dLdA0 = A0_tensor.grad.data.numpy()
        dLdW0 = torch_linear0.weight.grad.data.numpy()
        dLdb0 = torch_linear0.bias.grad.data.numpy().reshape(-1, 1)

        # Use your Linear layer to get the student answer
        mlp1.backward(dLdA2)
        dLdA0_ = mlp1.dLdA0
        dLdZ0_ = mlp1.dLdZ0
        dLdW0_ = mlp1.layers[0].dLdW
        dLdb0_ = mlp1.layers[0].dLdb
        dLdA1_ = mlp1.dLdA1
        dLdZ1_ = mlp1.dLdZ1
        dLdW1_ = mlp1.layers[2].dLdW
        dLdb1_ = mlp1.layers[2].dLdb
        
        if np.allclose(dLdZ1, dLdZ1_, atol=atol_threshold)==False:
            TEST_mlp1_dLdZ1=False

        if np.allclose(dLdA1, dLdA1_, atol=atol_threshold)==False:
            TEST_mlp1_dLdA1=False

        if np.allclose(dLdZ0, dLdZ0_, atol=atol_threshold)==False:
            TEST_mlp1_dLdZ0=False

        if np.allclose(dLdA0, dLdA0_, atol=atol_threshold)==False:
            TEST_mlp1_dLdA0=False

        if np.allclose(dLdW0, dLdW0_, atol=atol_threshold)==False:
            TEST_mlp1_dLdW0=False

        if np.allclose(dLdb0, dLdb0_, atol=atol_threshold)==False:
            TEST_mlp1_dLdb0=False


        if  TEST_mlp1_dLdZ1 and TEST_mlp1_dLdA1 and TEST_mlp1_dLdZ0  \
            and TEST_mlp1_dLdA0 and TEST_mlp1_dLdW0 and TEST_mlp1_dLdb0:
            print("Passed Backward on testcase", i + 1)
        else:
            print("Failed Backward on testcase", i + 1)
            print("A0:\n", A0, sep="")
            print("W0:\n", W0, sep="")
            print("b0:\n", b0, sep="")
            print("W1:\n", W1, sep="")
            print("b1:\n", b1, sep="")
            break

else:

    TEST_mlp1_Z0 = False
    TEST_mlp1_A1 = False
    TEST_mlp1_Z1 = False
    TEST_mlp1_A2 = False
    TEST_mlp1_dLdZ1 = False
    TEST_mlp1_dLdA1 = False
    TEST_mlp1_dLdZ0 = False
    TEST_mlp1_dLdA0 = False
    TEST_mlp1_dLdW0 = False
    TEST_mlp1_dLdb0 = False

"""
────────────────────────────────────────────────────────────────────────────────────
## MLP4
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_MLP4 = DEBUG_AND_GRADE_MLP4_flag

if DEBUG_AND_GRADE_MLP4:

    print("\n──────────────────────────────────────────")
    print("MLP4")
    print("──────────────────────────────────────────")

    for i in range(n_tests):
        TEST_mlp4_Z0 = True
        TEST_mlp4_A1 = True
        TEST_mlp4_Z1 = True
        TEST_mlp4_A2 = True
        TEST_mlp4_Z2 = True
        TEST_mlp4_A3 = True
        TEST_mlp4_Z3 = True
        TEST_mlp4_A4 = True
        TEST_mlp4_Z4 = True
        TEST_mlp4_A5 = True

        A0 = np.random.randn(4, 2).astype("f")
        W0 = np.random.randn(4, 2).astype("f")
        b0 = np.random.randn(4, ).astype("f")
        W1 = np.random.randn(8, 4).astype("f")
        b1 = np.random.randn(8, ).astype("f")
        W2 = np.random.randn(8, 8).astype("f")
        b2 = np.random.randn(8, ).astype("f")
        W3 = np.random.randn(4, 8).astype("f")
        b3 = np.random.randn(4, ).astype("f")
        W4 = np.random.randn(2, 4).astype("f")
        b4 = np.random.randn(2, ).astype("f")
        A0_tensor = torch.tensor(A0, requires_grad=True)

        # Use torch.linear to get the correct answer
        torch_linear0 = torch.nn.Linear(2, 4)
        torch_linear0.weight.data = torch.tensor(W0)
        torch_linear0.bias.data = torch.tensor(b0)
        torch_linear0.requires_grad_()
        torch_linear1 = torch.nn.Linear(4, 8)
        torch_linear1.weight.data = torch.tensor(W1)
        torch_linear1.bias.data = torch.tensor(b1)
        torch_linear1.requires_grad_()
        torch_linear2 = torch.nn.Linear(8, 8)
        torch_linear2.weight.data = torch.tensor(W2)
        torch_linear2.bias.data = torch.tensor(b2)
        torch_linear2.requires_grad_()
        torch_linear3 = torch.nn.Linear(8, 4)
        torch_linear3.weight.data = torch.tensor(W3)
        torch_linear3.bias.data = torch.tensor(b3)
        torch_linear3.requires_grad_()
        torch_linear4 = torch.nn.Linear(4, 2)
        torch_linear4.weight.data = torch.tensor(W4)
        torch_linear4.bias.data = torch.tensor(b4)
        torch_linear4.requires_grad_()
        Z0_tensor = torch_linear0(A0_tensor)
        Z0 = Z0_tensor.detach().numpy()
        A1_tensor = torch.relu(Z0_tensor)
        A1 = A1_tensor.detach().numpy()
        A1_tensor_copy = torch.tensor(A1, requires_grad=True)
        Z1_tensor = torch_linear1(A1_tensor_copy)
        Z1 = Z1_tensor.detach().numpy()
        A2_tensor = torch.relu(Z1_tensor)
        A2 = A2_tensor.detach().numpy()
        A2_tensor_copy = torch.tensor(A2, requires_grad=True)
        Z2_tensor = torch_linear2(A2_tensor_copy)
        Z2 = Z2_tensor.detach().numpy()
        A3_tensor = torch.relu(Z2_tensor)
        A3 = A3_tensor.detach().numpy()
        A3_tensor_copy = torch.tensor(A3, requires_grad=True)
        Z3_tensor = torch_linear3(A3_tensor_copy)
        Z3 = Z3_tensor.detach().numpy()
        A4_tensor = torch.relu(Z3_tensor)
        A4 = A4_tensor.detach().numpy()
        A4_tensor_copy = torch.tensor(A4, requires_grad=True)
        Z4_tensor = torch_linear4(A4_tensor_copy)
        Z4 = Z4_tensor.detach().numpy()
        A5_tensor = torch.relu(Z4_tensor)
        A5 = A5_tensor.detach().numpy()

        # Use your Linear layer to get the student answer
        mlp4 = MLP4(debug=True)

        if not (mlp4.layers[0].W.shape == W0.shape and mlp4.layers[2].W.shape == W1.shape and mlp4.layers[4].W.shape == W2.shape
            and mlp4.layers[6].W.shape == W3.shape and mlp4.layers[8].W.shape == W4.shape):
            print ("Incorrect architecture")
            TEST_mlp4_Z0 = False
            TEST_mlp4_A1 = False
            TEST_mlp4_Z1 = False
            TEST_mlp4_A2 = False
            TEST_mlp4_Z2 = False
            TEST_mlp4_A3 = False
            TEST_mlp4_Z3 = False
            TEST_mlp4_A4 = False
            TEST_mlp4_Z4 = False
            TEST_mlp4_A5 = False
            break

        mlp4.layers[0].W = W0
        mlp4.layers[0].b = b0.reshape(-1, 1)
        mlp4.layers[2].W = W1
        mlp4.layers[2].b = b1.reshape(-1, 1)
        mlp4.layers[4].W = W2
        mlp4.layers[4].b = b2.reshape(-1, 1)
        mlp4.layers[6].W = W3
        mlp4.layers[6].b = b3.reshape(-1, 1)
        mlp4.layers[8].W = W4
        mlp4.layers[8].b = b4.reshape(-1, 1)
        A5_ = mlp4.forward(A0)
        Z0_ = mlp4.A[1]
        Z1_ = mlp4.A[3]
        A1_ = mlp4.A[2]
        Z2_ = mlp4.A[5]
        A2_ = mlp4.A[4]
        Z3_ = mlp4.A[7]
        A3_ = mlp4.A[6]
        Z4_ = mlp4.A[9]
        A4_ = mlp4.A[8]


        # Compare the student answer with the correct answer

        if np.allclose(Z0, Z0_, atol=atol_threshold)==False:
            TEST_mlp4_Z0=False

        if np.allclose(A1, A1_, atol=atol_threshold)==False:
            TEST_mlp4_A1=False

        if np.allclose(Z1, Z1_, atol=atol_threshold)==False:
            TEST_mlp4_Z1=False

        if np.allclose(A2, A2_, atol=atol_threshold)==False:
            TEST_mlp4_A2=False

        if np.allclose(Z2, Z2_, atol=atol_threshold)==False:
            TEST_mlp4_Z2=False

        if np.allclose(A3, A3_, atol=atol_threshold)==False:
            TEST_mlp4_A3=False

        if np.allclose(Z3, Z3_, atol=atol_threshold)==False:
            TEST_mlp4_Z3=False

        if np.allclose(A4, A4_, atol=atol_threshold)==False:
            TEST_mlp4_A4=False

        if np.allclose(Z4, Z4_, atol=atol_threshold)==False:
            TEST_mlp4_Z4=False

        if np.allclose(A5, A5_, atol=atol_threshold)==False:
            TEST_mlp4_A5=False

        if TEST_mlp4_Z0 and TEST_mlp4_A1 and TEST_mlp4_Z1 and TEST_mlp4_A2 and TEST_mlp4_Z2 \
            and TEST_mlp4_A3 and TEST_mlp4_Z3 and TEST_mlp4_A4 and TEST_mlp4_Z4 and TEST_mlp4_A5:
            print("Passed Forward on testcase", i + 1)

        else:
            print("Failed Forward on testcase", i + 1)
            print("A0:\n", A0, sep="")
            print("W0:\n", W0, sep="")
            print("b0:\n", b0, sep="")
            print("W1:\n", W1, sep="")
            print("b1:\n", b1, sep="")
            print("W2:\n", W2, sep="")
            print("b2:\n", b2, sep="")
            print("W3:\n", W3, sep="")
            print("b3:\n", b3, sep="")
            print("W4:\n", W4, sep="")
            print("b4:\n", b4, sep="")
            break

        # test backward
        TEST_mlp4_dLdZ0 = True
        TEST_mlp4_dLdA0 = True
        TEST_mlp4_dLdW0 = True
        TEST_mlp4_dLdb0 = True
        TEST_mlp4_dLdZ1 = True
        TEST_mlp4_dLdA1 = True
        TEST_mlp4_dLdW1 = True
        TEST_mlp4_dLdb1 = True
        TEST_mlp4_dLdZ2 = True
        TEST_mlp4_dLdA2 = True
        TEST_mlp4_dLdW2 = True
        TEST_mlp4_dLdb2 = True
        TEST_mlp4_dLdZ3 = True
        TEST_mlp4_dLdA3 = True
        TEST_mlp4_dLdW3 = True
        TEST_mlp4_dLdb3 = True
        TEST_mlp4_dLdZ4 = True
        TEST_mlp4_dLdA4 = True
        TEST_mlp4_dLdW4 = True
        TEST_mlp4_dLdb4 = True

        # Use torch.linear to get the correct answer
        dLdA5 = np.random.randn(4, 2).astype("f")
        dLdA5_tensor = torch.tensor(dLdA5)
        dA5dZ4 = torch.autograd.grad(A5_tensor, Z4_tensor, grad_outputs=torch.ones_like(A5_tensor))[0].numpy()
        dLdZ4 = dLdA5*dA5dZ4
        Z4_tensor.backward(gradient=torch.tensor(dLdZ4), retain_graph=True)
        dLdA4 = A4_tensor_copy.grad.data.numpy()
        dLdW4 = torch_linear4.weight.grad.data.numpy()
        dLdb4 = torch_linear4.bias.grad.data.numpy().reshape(-1, 1)
        dA4dZ3 = torch.autograd.grad(A4_tensor, Z3_tensor, grad_outputs=torch.ones_like(A4_tensor))[0].numpy()
        dLdZ3 = dLdA4*dA4dZ3
        Z3_tensor.backward(gradient=torch.tensor(dLdZ3), retain_graph=True)
        dLdA3 = A3_tensor_copy.grad.data.numpy()
        dLdW3 = torch_linear3.weight.grad.data.numpy()
        dLdb3 = torch_linear3.bias.grad.data.numpy().reshape(-1, 1)
        dA3dZ2 = torch.autograd.grad(A3_tensor, Z2_tensor, grad_outputs=torch.ones_like(A3_tensor))[0].numpy()
        dLdZ2 = dLdA3*dA3dZ2
        Z2_tensor.backward(gradient=torch.tensor(dLdZ2), retain_graph=True)
        dLdA2 = A2_tensor_copy.grad.data.numpy()
        dLdW2 = torch_linear2.weight.grad.data.numpy()
        dLdb2 = torch_linear2.bias.grad.data.numpy().reshape(-1, 1)
        dA2dZ1 = torch.autograd.grad(A2_tensor, Z1_tensor, grad_outputs=torch.ones_like(A2_tensor))[0].numpy()
        dLdZ1 = dLdA2*dA2dZ1
        Z1_tensor.backward(gradient=torch.tensor(dLdZ1), retain_graph=True)
        dLdA1 = A1_tensor_copy.grad.data.numpy()
        dLdW1 = torch_linear1.weight.grad.data.numpy()
        dLdb1 = torch_linear1.bias.grad.data.numpy().reshape(-1, 1)
        dA1dZ0 = torch.autograd.grad(A1_tensor, Z0_tensor, grad_outputs=torch.ones_like(A1_tensor))[0].numpy()
        dLdZ0 = dLdA1*dA1dZ0
        Z0_tensor.backward(gradient=torch.tensor(dLdZ0), retain_graph=True)
        dLdA0 = A0_tensor.grad.data.numpy()
        dLdW0 = torch_linear0.weight.grad.data.numpy()
        dLdb0 = torch_linear0.bias.grad.data.numpy().reshape(-1, 1)

        # Use your Linear layer to get the student answer
        mlp4.backward(dLdA5)
        if np.allclose(dLdZ4, mlp4.dLdA[9], atol=atol_threshold)==False:
            TEST_mlp4_dLdZ4=False

        if np.allclose(dLdA4, mlp4.dLdA[8], atol=atol_threshold)==False:
            TEST_mlp4_dLdA4=False

        if np.allclose(dLdW4, mlp4.layers[8].dLdW, atol=atol_threshold)==False:
            TEST_mlp4_dLdW4=False

        if np.allclose(dLdb4, mlp4.layers[8].dLdb, atol=atol_threshold)==False:
            TEST_mlp4_dLdb4=False

        if np.allclose(dLdZ3, mlp4.dLdA[7], atol=atol_threshold)==False:
            TEST_mlp4_dLdZ3=False

        if np.allclose(dLdA3, mlp4.dLdA[6], atol=atol_threshold)==False:
            TEST_mlp4_dLdA3=False

        if np.allclose(dLdW3, mlp4.layers[6].dLdW, atol=atol_threshold)==False:
            TEST_mlp4_dLdW3=False

        if np.allclose(dLdb3, mlp4.layers[6].dLdb, atol=atol_threshold)==False:
            TEST_mlp4_dLdb3=False

        if np.allclose(dLdZ2, mlp4.dLdA[5], atol=atol_threshold)==False:
            TEST_mlp4_dLdZ2=False

        if np.allclose(dLdA2, mlp4.dLdA[4], atol=atol_threshold)==False:
            TEST_mlp4_dLdA2=False

        if np.allclose(dLdW2, mlp4.layers[4].dLdW, atol=atol_threshold)==False:
            TEST_mlp4_dLdW2=False

        if np.allclose(dLdb2, mlp4.layers[4].dLdb, atol=atol_threshold)==False:
            TEST_mlp4_dLdb2=False

        if np.allclose(dLdZ1, mlp4.dLdA[3], atol=atol_threshold)==False:
            TEST_mlp4_dLdZ1=False

        if np.allclose(dLdA1, mlp4.dLdA[2], atol=atol_threshold)==False:
            TEST_mlp4_dLdA1=False

        if np.allclose(dLdW1, mlp4.layers[2].dLdW, atol=atol_threshold)==False:
            TEST_mlp4_dLdW1=False

        if np.allclose(dLdb1, mlp4.layers[2].dLdb, atol=atol_threshold)==False:
            TEST_mlp4_dLdb1=False

        if np.allclose(dLdZ0, mlp4.dLdA[1], atol=atol_threshold)==False:
            TEST_mlp4_dLdZ0=False

        if np.allclose(dLdA0, mlp4.dLdA[0], atol=atol_threshold)==False:
            TEST_mlp4_dLdA0=False

        if np.allclose(dLdW0, mlp4.layers[0].dLdW, atol=atol_threshold)==False:
            TEST_mlp4_dLdW0=False

        if np.allclose(dLdb0, mlp4.layers[0].dLdb, atol=atol_threshold)==False:
            TEST_mlp4_dLdb0=False

        if  TEST_mlp4_dLdZ0 and TEST_mlp4_dLdA0 and TEST_mlp4_dLdW0 and TEST_mlp4_dLdb0 \
            and TEST_mlp4_dLdZ1 and TEST_mlp4_dLdA1 and TEST_mlp4_dLdW1 and TEST_mlp4_dLdb1 \
            and TEST_mlp4_dLdZ2 and TEST_mlp4_dLdA2 and TEST_mlp4_dLdW2 and TEST_mlp4_dLdb2 \
            and TEST_mlp4_dLdZ3 and TEST_mlp4_dLdA3 and TEST_mlp4_dLdW3 and TEST_mlp4_dLdb3 \
            and TEST_mlp4_dLdZ4 and TEST_mlp4_dLdA4 and TEST_mlp4_dLdW4 and TEST_mlp4_dLdb4:
            print("Passed Backward on testcase", i + 1)

        else:
            print("Failed Backward on testcase", i + 1)
            print("A0:\n", A0, sep="")
            print("W0:\n", W0, sep="")
            print("b0:\n", b0, sep="")
            print("W1:\n", W1, sep="")
            print("b1:\n", b1, sep="")
            print("W2:\n", W2, sep="")
            print("b2:\n", b2, sep="")
            print("W3:\n", W3, sep="")
            print("b3:\n", b3, sep="")
            print("W4:\n", W4, sep="")
            print("b4:\n", b4, sep="")
            break


else:

    TEST_mlp4_Z0 = False
    TEST_mlp4_A1 = False
    TEST_mlp4_Z1 = False
    TEST_mlp4_A2 = False
    TEST_mlp4_Z2 = False
    TEST_mlp4_A3 = False
    TEST_mlp4_Z3 = False
    TEST_mlp4_A4 = False
    TEST_mlp4_Z4 = False
    TEST_mlp4_A5 = False

    TEST_mlp4_dLdZ0 = False
    TEST_mlp4_dLdA0 = False
    TEST_mlp4_dLdW0 = False
    TEST_mlp4_dLdb0 = False
    TEST_mlp4_dLdZ1 = False
    TEST_mlp4_dLdA1 = False
    TEST_mlp4_dLdW1 = False
    TEST_mlp4_dLdb1 = False
    TEST_mlp4_dLdZ2 = False
    TEST_mlp4_dLdA2 = False
    TEST_mlp4_dLdW2 = False
    TEST_mlp4_dLdb2 = False
    TEST_mlp4_dLdZ3 = False
    TEST_mlp4_dLdA3 = False
    TEST_mlp4_dLdW3 = False
    TEST_mlp4_dLdb3 = False
    TEST_mlp4_dLdZ4 = False
    TEST_mlp4_dLdA4 = False
    TEST_mlp4_dLdW4 = False
    TEST_mlp4_dLdb4 = False



"""
────────────────────────────────────────────────────────────────────────────────────
# Loss
────────────────────────────────────────────────────────────────────────────────────
"""

"""
────────────────────────────────────────────────────────────────────────────────────
## MSELoss
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_MSELOSS = DEBUG_AND_GRADE_MSELOSS_flag

if DEBUG_AND_GRADE_MSELOSS:

    print("\n──────────────────────────────────────────")
    print("MSELoss | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    Y = np.array([
        [0., 1.],
        [1., 0.],
        [1., 0.],
        [0., 1.]], dtype="f")

    mse = MSELoss()

    L = mse.forward(A, Y)
    print("\nL =\n", L.round(4), sep="")

    dLdA = mse.backward()
    print("\ndLdA =\n", dLdA, sep="")

    print("\n──────────────────────────────────────────")
    print("MSELOSS | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    L_solution = np.array(6.5, dtype="f")

    dLdA_solution = np.array([
        [-0.5, -0.5],
        [-0.375, -0.125],
        [-0.125, 0.125],
        [0.25, 0.25]], dtype="f")*2

    print("\nL =\n", L_solution, "\n", sep="")
    print("\ndLdA =\n", dLdA_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("MSELOSS | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("\n           Pass?")

    TEST_mseloss_L = np.allclose(L.round(4), L_solution, atol=atol_threshold)
    print("Test L:   ", TEST_mseloss_L)

    TEST_mseloss_dLdA = np.allclose(dLdA.round(4), dLdA_solution, atol=atol_threshold)
    print("Test dLdA:", TEST_mseloss_dLdA)

else:

    TEST_mseloss_L = False
    TEST_mseloss_dLdA = False

"""
────────────────────────────────────────────────────────────────────────────────────
## CrossEntropyLoss
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_CROSSENTROPYLOSS = DEBUG_AND_GRADE_CROSSENTROPYLOSS_flag

if DEBUG_AND_GRADE_CROSSENTROPYLOSS:

    print("\n──────────────────────────────────────────")
    print("CROSSENTROPYLOSS | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [-4., -3.],
        [-2., -1.],
        [0., 1.],
        [2., 3.]], dtype="f")

    Y = np.array([
        [0., 1.],
        [1., 0.],
        [1., 0.],
        [0., 1.]], dtype="f")

    xent = CrossEntropyLoss()

    L = xent.forward(A, Y)
    print("\nL =\n", L.round(4), sep="")

    dLdA = xent.backward()
    print("\ndLdA =\n", dLdA, sep="")

    print("──────────────────────────────────────────")
    print("CROSSENTROPYLOSS | SOLUTION OUTPUT\n")
    print("──────────────────────────────────────────")

    L_solution = np.array(0.8133, dtype="f")

    dLdA_solution = np.array([
        [0.2689, -0.2689],
        [-0.7311, 0.7311],
        [-0.7311, 0.7311],
        [0.2689, -0.2689]], dtype="f")/4

    print("\nL =\n", L_solution, sep="")

    print("\ndLdA =\n", dLdA_solution, sep="")

    print("\n──────────────────────────────────────────")
    print("CROSSENTROPYLOSS | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("           Pass?")

    TEST_crossentropyloss_L = np.allclose(L.round(4), L_solution, atol=atol_threshold)
    print("Test L:   ", TEST_crossentropyloss_L)

    TEST_crossentropyloss_dLdA = np.allclose(dLdA.round(4), dLdA_solution, atol=atol_threshold)
    print("Test dLdA:", TEST_crossentropyloss_dLdA)

else:

    TEST_crossentropyloss_L = False
    TEST_crossentropyloss_dLdA = False


"""
────────────────────────────────────────────────────────────────────────────────────
# SGD
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_SGD = DEBUG_AND_GRADE_SGD_flag

if DEBUG_AND_GRADE_SGD:

    print("\n──────────────────────────────────────────")
    print("SGD | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    class PseudoModel:
        def __init__(self):
            self.layers = [Linear(3, 2)]
            self.f = [ReLU()]

        def forward(self, A):
            return NotImplemented

        def backward(self):
            return NotImplemented

    # Create Example Model
    pseudo_model = PseudoModel()

    pseudo_model.layers[0].W = np.ones((3, 2))
    pseudo_model.layers[0].dLdW = np.ones((3, 2)) / 10
    pseudo_model.layers[0].b = np.ones((3, 1))
    pseudo_model.layers[0].dLdb = np.ones((3, 1)) / 10

    print("\nInitialized Parameters:\n")
    print("W =\n", pseudo_model.layers[0].W, "\n", sep="")
    print("b =\n", pseudo_model.layers[0].b, "\n", sep="")

    # Test Example Models
    optimizer = SGD(pseudo_model, lr=0.9)
    optimizer.step()

    print("Parameters After SGD (Step=1)\n")

    W_1 = pseudo_model.layers[0].W.copy()
    b_1 = pseudo_model.layers[0].b.copy()
    print("W =\n", W_1, "\n", sep="")
    print("b =\n", b_1, "\n", sep="")

    optimizer.step()

    print("Parameters After SGD (Step=2)\n")

    W_2 = pseudo_model.layers[0].W
    b_2 = pseudo_model.layers[0].b
    print("W =\n", W_2, "\n", sep="")
    print("b =\n", b_2, "\n", sep="")

    print("──────────────────────────────────────────")
    print("SGD | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    W_1_solution = np.array([
        [0.91, 0.91],
        [0.91, 0.91],
        [0.91, 0.91]], dtype="f")

    b_1_solution = np.array([
        [0.91],
        [0.91],
        [0.91]], dtype="f")

    W_2_solution = np.array([
        [0.82, 0.82],
        [0.82, 0.82],
        [0.82, 0.82]], dtype="f")

    b_2_solution = np.array([
        [0.82],
        [0.82],
        [0.82]], dtype="f")

    print("\nParameters After SGD (Step=1)\n")

    print("W =\n", W_1_solution, "\n", sep="")
    print("b =\n", b_1_solution, "\n", sep="")

    print("Parameters After SGD (Step=2)\n")

    print("W =\n", W_2_solution, "\n", sep="")
    print("b =\n", b_2_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("SGD | TEST RESULTS")
    print("──────────────────────────────────────────")

    print("                 Pass?")

    TEST_sgd_W_1 = np.allclose(W_1.round(4), W_1_solution, atol=atol_threshold)
    print("Test W (Step 1):", TEST_sgd_W_1)

    TEST_sgd_b_1 = np.allclose(b_1.round(4), b_1_solution, atol=atol_threshold)
    print("Test b (Step 1):", TEST_sgd_b_1)

    TEST_sgd_W_2 = np.allclose(W_2.round(4), W_2_solution, atol=atol_threshold)
    print("Test W (Step 2):", TEST_sgd_W_2)

    TEST_sgd_b_2 = np.allclose(b_2.round(4), b_2_solution, atol=atol_threshold)
    print("Test b (Step 2):", TEST_sgd_b_2)

else:

    TEST_sgd_W_1 = False
    TEST_sgd_b_1 = False
    TEST_sgd_W_2 = False
    TEST_sgd_b_2 = False

"""
────────────────────────────────────────────────────────────────────────────────────
# SGD (With Momentum)
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_SGD_M = DEBUG_AND_GRADE_SGD_momentum_flag

if DEBUG_AND_GRADE_SGD_M:

    print("\n──────────────────────────────────────────")
    print("SGD with Momentum | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    class PseudoModel:
        def __init__(self):
            self.layers = [Linear(3, 2)]
            self.f = [ReLU()]

        def forward(self, A):
            return NotImplemented

        def backward(self):
            return NotImplemented

    # Create Example Model
    pseudo_model = PseudoModel()

    pseudo_model.layers[0].W = np.ones((3, 2))
    pseudo_model.layers[0].dLdW = np.ones((3, 2)) / 10
    pseudo_model.layers[0].b = np.ones((3, 1))
    pseudo_model.layers[0].dLdb = np.ones((3, 1)) / 10

    print("\nInitialized Parameters:\n")
    print("W =\n", pseudo_model.layers[0].W, "\n", sep="")
    print("b =\n", pseudo_model.layers[0].b, "\n", sep="")

    # Test Example Models
    optimizer = SGD(pseudo_model, lr=0.9, momentum=0.9)
    optimizer.step()

    print("Parameters After SGD (Step=1)\n")

    W_1 = pseudo_model.layers[0].W.copy()
    b_1 = pseudo_model.layers[0].b.copy()
    print("W =\n", W_1, "\n", sep="")
    print("b =\n", b_1, "\n", sep="")

    optimizer.step()

    print("Parameters After SGD (Step=2)\n")

    W_2 = pseudo_model.layers[0].W
    b_2 = pseudo_model.layers[0].b
    print("W =\n", W_2, "\n", sep="")
    print("b =\n", b_2, "\n", sep="")

    print("──────────────────────────────────────────")
    print("SGD with Momentum | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    W_1_solution = np.array([
        [0.91, 0.91],
        [0.91, 0.91],
        [0.91, 0.91]], dtype="f")

    b_1_solution = np.array([
        [0.91],
        [0.91],
        [0.91]], dtype="f")

    W_2_solution = np.array([
        [0.739, 0.739],
        [0.739, 0.739],
        [0.739, 0.739]], dtype="f")

    b_2_solution = np.array([
        [0.739],
        [0.739],
        [0.739]], dtype="f")

    print("\nParameters After SGD (Step=1)\n")

    print("W =\n", W_1_solution, "\n", sep="")
    print("b =\n", b_1_solution, "\n", sep="")

    print("Parameters After SGD (Step=2)\n")

    print("W =\n", W_2_solution, "\n", sep="")
    print("b =\n", b_2_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("SGD with Momentum| TEST RESULTS")
    print("──────────────────────────────────────────")

    print("                 Pass?")

    TEST_sgd_W_m_1 = np.allclose(W_1.round(4), W_1_solution, atol=atol_threshold)
    print("Test W (Step 1):", TEST_sgd_W_m_1)

    TEST_sgd_b_m_1 = np.allclose(b_1.round(4), b_1_solution, atol=atol_threshold)
    print("Test b (Step 1):", TEST_sgd_b_m_1)

    TEST_sgd_W_m_2 = np.allclose(W_2.round(4), W_2_solution, atol=atol_threshold)
    print("Test W (Step 2):", TEST_sgd_W_m_2)

    TEST_sgd_b_m_2 = np.allclose(b_2.round(4), b_2_solution, atol=atol_threshold)
    print("Test b (Step 2):", TEST_sgd_b_m_2)

else:

    TEST_sgd_W_m_1 = False
    TEST_sgd_b_m_1 = False
    TEST_sgd_W_m_2 = False
    TEST_sgd_b_m_2 = False


"""
────────────────────────────────────────────────────────────────────────────────────
## BATCH NORMALIZATION
────────────────────────────────────────────────────────────────────────────────────
"""

DEBUG_AND_GRADE_BATCHNORM = DEBUG_AND_GRADE_BATCHNORM_flag

if DEBUG_AND_GRADE_BATCHNORM:

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Eval) | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    A = np.array([
        [1., 4.],
        [7., 0.],
        [1., 0.],
        [7., 4.]], dtype="f")

    BW = np.array([
        [2., 5.]], dtype="f")

    Bb = np.array([
        [-1., 2.]], dtype="f")

    bn = BatchNorm1d(2)
    bn.BW = BW
    bn.Bb = Bb

    BZ = bn.forward(A, eval=True)
    print("\n(eval) BZ =\n", BZ, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Eval) | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    BZ_solution = np.array([
        [1., 22.],
        [13., 2.],
        [1., 2.],
        [13., 22.]], dtype="f")

    print("\n(eval) BZ =\n", BZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Eval) | TEST RESULTS")
    print("──────────────────────────────────────────")

    TEST_batchnorm_eval_BZ = np.allclose(BZ.round(4), BZ_solution, atol=atol_threshold)
    print("\nTest (eval) BZ: ", TEST_batchnorm_eval_BZ, "\n", sep="")

else:

    TEST_batchnorm_eval_BZ = False

if DEBUG_AND_GRADE_BATCHNORM:

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Train) | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    bn = BatchNorm1d(2)
    bn.BW = BW
    bn.Bb = Bb

    BZ = bn.forward(A, eval=False)
    print("\n(train) BZ =\n", BZ, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Train) | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    BZ_solution = np.array([
        [-3., 7.],
        [1., -3.],
        [-3., -3.],
        [1., 7.]])

    print("\n(train) BZ =\n", BZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM FORWARD (Train) | TEST RESULTS")
    print("──────────────────────────────────────────")

    TEST_batchnorm_train_BZ = np.allclose(BZ.round(4), BZ_solution, atol=atol_threshold)
    print("\nTest (train) BZ: ", TEST_batchnorm_train_BZ, "\n", sep="")

else:

    TEST_batchnorm_train_BZ = False

if DEBUG_AND_GRADE_BATCHNORM:

    print("\n──────────────────────────────────────────")
    print("BATCHNORM BACKWARD | STUDENT OUTPUT")
    print("──────────────────────────────────────────")

    dLdA = np.array([
        [-6., 2.],
        [-12., 16.],
        [-12., 20.],
        [-6., 2.]], dtype="f")

    dLdZ = bn.backward(dLdA)
    print("\ndLdZ =\n", dLdZ, sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM BACKWARD | SOLUTION OUTPUT")
    print("──────────────────────────────────────────")

    dLdZ_solution = np.array([
        [2., 0.],
        [-2., -5.],
        [-2., 5.],
        [2., 0.]], dtype="f")

    print("\ndLdZ =\n", dLdZ_solution, "\n", sep="")

    print("\n──────────────────────────────────────────")
    print("BATCHNORM BACKWARD | TEST RESULTS")
    print("──────────────────────────────────────────")

    TEST_batchnorm_dLdZ = np.allclose(dLdZ.round(4), dLdZ_solution, atol=atol_threshold)
    print("\nTest dLdZ: ", TEST_batchnorm_dLdZ, "\n", sep="")

else:

    TEST_batchnorm_dLdZ = False


"""
────────────────────────────────────────────────────────────────────────────────────
## SCORE AND GRADE TESTS
────────────────────────────────────────────────────────────────────────────────────
"""

TEST_activations = (
    TEST_identity_A and
    TEST_identity_dLdZ and
    TEST_sigmoid_A and
    TEST_sigmoid_dLdZ and
    TEST_tanh_A and
    TEST_tanh_dLdZ and
    TEST_relu_A and
    TEST_relu_dLdZ and
    TEST_gelu_A and
    TEST_gelu_dLdZ and
    TEST_softmax_A and
    TEST_softmax_dLdZ)

TEST_loss = (
    TEST_mseloss_L and
    TEST_mseloss_dLdA and
    TEST_crossentropyloss_L and
    TEST_crossentropyloss_dLdA)

TEST_linear = (
    TEST_linear_Z and
    TEST_linear_dLdA and
    TEST_linear_dLdW and
    TEST_linear_dLdb)

TEST_sgd = (
    TEST_sgd_W_m_1 and
    TEST_sgd_b_m_1 and
    TEST_sgd_W_m_2 and
    TEST_sgd_b_m_2)

TEST_mlp0 = (
    TEST_mlp0_Z0 and
    TEST_mlp0_A1 and
    TEST_mlp0_dLdZ0 and
    TEST_mlp0_dLdA0 and
    TEST_mlp0_dLdW0 and
    TEST_mlp0_dLdb0)

TEST_mlp1 = (
    TEST_mlp1_Z0 and
    TEST_mlp1_A1 and
    TEST_mlp1_Z1 and
    TEST_mlp1_A2 and
    TEST_mlp1_dLdZ1 and
    TEST_mlp1_dLdA1 and
    TEST_mlp1_dLdZ0 and
    TEST_mlp1_dLdA0 and
    TEST_mlp1_dLdW0 and
    TEST_mlp1_dLdb0)

TEST_mlp4 = (
    TEST_mlp4_Z0 and
    TEST_mlp4_A1 and
    TEST_mlp4_Z1 and
    TEST_mlp4_A2 and
    TEST_mlp4_Z2 and
    TEST_mlp4_A3 and
    TEST_mlp4_Z3 and
    TEST_mlp4_A4 and
    TEST_mlp4_Z4 and
    TEST_mlp4_A5 and
    TEST_mlp4_dLdZ0 and
    TEST_mlp4_dLdA0 and
    TEST_mlp4_dLdW0 and
    TEST_mlp4_dLdb0 and
    TEST_mlp4_dLdZ1 and
    TEST_mlp4_dLdA1 and
    TEST_mlp4_dLdW1 and
    TEST_mlp4_dLdb1 and
    TEST_mlp4_dLdZ2 and
    TEST_mlp4_dLdA2 and
    TEST_mlp4_dLdW2 and
    TEST_mlp4_dLdb2 and
    TEST_mlp4_dLdZ3 and
    TEST_mlp4_dLdA3 and
    TEST_mlp4_dLdW3 and
    TEST_mlp4_dLdb3 and
    TEST_mlp4_dLdZ4 and
    TEST_mlp4_dLdA4 and
    TEST_mlp4_dLdW4 and
    TEST_mlp4_dLdb4)

TEST_batchnorm = (
    TEST_batchnorm_eval_BZ and
    TEST_batchnorm_train_BZ and
    TEST_batchnorm_dLdZ)

SCORE_LOGS = {
    "Linear Layer": 15 * int(TEST_linear),
    "Activation": 10 * int(TEST_activations),
    "MLP0": 10 * int(TEST_mlp0),
    "MLP1": 10 * int(TEST_mlp1),
    "MLP4": 15 * int(TEST_mlp4),
    "Loss": 10 * int(TEST_loss),
    "SGD": 10 * int(TEST_sgd),
    "Batch Norm": 20 * int(TEST_batchnorm)
}


print("\n")
print("TEST   | STATUS | POINTS | DESCRIPTION")
print("───────┼────────┼────────┼────────────────────────────────")

for i, (key, value) in enumerate(SCORE_LOGS.items()):

    index_str = str(i).zfill(1)
    point_str = str(value).zfill(2) + "     │ "

    if value == 0:
        status_str = " │ FAILED │ "
    else:
        status_str = " │ PASSED │ "

    print("Test ", index_str, status_str, point_str, key, sep="")

print("\n")

"""
────────────────────────────────────────────────────────────────────────────────────
## FINAL AUTOLAB SCORES
────────────────────────────────────────────────────────────────────────────────────
"""

print(json.dumps({'scores': SCORE_LOGS}))