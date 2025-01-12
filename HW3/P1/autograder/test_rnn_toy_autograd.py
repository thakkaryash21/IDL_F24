import sys, pdb, os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from test import Test
sys.path.append("./")
from mytorch import autograd_engine
from mytorch.rnn_cell import *
from mytorch.nn.loss import *
from models.rnn_classifier import *

def print_gradients(grad):
  print(f"grad: {grad}")

# Reference Pytorch RNN Model
class ReferenceModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, rnn_layers=2):
        super(ReferenceModel, self).__init__()
        self.rnn = nn.RNN(input_size,
                          hidden_size,
                          num_layers=rnn_layers,
                          bias=True,
                          batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x, init_h=None):
        out, hidden = self.rnn(x, init_h)
        out = self.output(out[:, -1, :])
        return out


class RNNToyTest(Test):
    def __init__(self):
        pass

    def test_rnncell_forward(self):
        np.random.seed(11785)
        torch.manual_seed(11785)
        # Using i within this loop to vary the inputs
        i = 1

        # Make pytorch rnn cell and get weights
        pytorch_rnn_cell = nn.RNNCell(i * 2, i * 3)
        state_dict = pytorch_rnn_cell.state_dict()
        W_ih, W_hh = state_dict["weight_ih"].numpy(), state_dict["weight_hh"].numpy()
        b_ih, b_hh = state_dict["bias_ih"].numpy(), state_dict["bias_hh"].numpy()

        # Set user cell and weights
        # NOTE: Autograd object must be instantiated and passed
        # Besides that everything else for this test case is 
        # equivalent to the non-Autograd version
        autograd = autograd_engine.Autograd()
        user_cell = RNNCell(i * 2, i * 3, autograd)
        user_cell.init_weights(W_ih, W_hh, b_ih, b_hh)

        # Get inputs
        time_steps = i * 2
        inp = torch.randn(time_steps, i * 2, i * 2)
        hx = torch.randn(i * 2, i * 3)
        inp_user = inp.numpy()
        hx_user = hx.numpy()

        # Loop through inputs
        for t in range(time_steps):
            print('*** time step {} ***'.format(t))
            print('input: \n{} \nhidden: \n{}'.format(inp_user[t],hx.detach().numpy()))
            hx = pytorch_rnn_cell(inp[t], hx)
            hx_user = user_cell(inp_user[t], hx_user)
            if not np.allclose(hx.detach().numpy(), hx_user, rtol=1e-03):
                print(f'wrong value for h_prime in rnn cell forward at timestep {t}\n')
                return False
            print('*** passed ***')

        return True

    def test_rnncell_backward(self):

        # NOTE: I recommend changing this testcase to more effectively test the autograd implementation.
        # After looping through each timestep to get model output, a dummy loss is created by summing the 
        # final output and backpropagating the loss. The input, hidden and network parameter gradients are
        # then compared to the official pytorch implementation.

        np.random.seed(11785)
        torch.manual_seed(11785)

        # Using i within this loop to vary the inputs
        i = 1

        # Make pytorch rnn cell and get weights
        pytorch_rnn_cell = nn.RNNCell(i * 2, i * 3)
        state_dict = pytorch_rnn_cell.state_dict()
        W_ih, W_hh = state_dict["weight_ih"].numpy(), state_dict["weight_hh"].numpy()
        b_ih, b_hh = state_dict["bias_ih"].numpy(), state_dict["bias_hh"].numpy()

        autograd = autograd_engine.Autograd()
        # Set user cell and weights
        user_cell = RNNCell(i * 2, i * 3, autograd)
        user_cell.init_weights(W_ih, W_hh, b_ih, b_hh)

        # Get inputs
        time_steps = i * 2
        inp = torch.randn(time_steps, i * 2, i * 2, requires_grad=True)
        inp_user = inp.detach().numpy()
        pytorch_hiddens = [torch.randn(i * 2, i * 3, requires_grad=True)]
        user_hiddens = [pytorch_hiddens[0].detach().numpy()]

        # Loop through inputs
        for t in range(time_steps):
            hx = pytorch_rnn_cell(inp[t], pytorch_hiddens[-1])

            # Slice input and store in computational graph
            idx = np.index_exp[t]
            inp_user_slice = inp_user[idx].copy()
            autograd.add_operation(inputs=[inp_user, np.array(idx, dtype=object)],
                                   output=inp_user_slice, 
                                   gradients_to_update=[None, None],
                                   backward_operation=slice_backward)
            hx_user = user_cell(inp_user_slice, user_hiddens[-1])

            pytorch_hiddens.append(hx)
            user_hiddens.append(hx_user)
            if not np.allclose(hx.detach().numpy(), hx_user, rtol=1e-03):
                print(f'wrong value for h_prime in rnn cell forward at timestep {t}\n')
                return False

        # pytorch rnncell backward
        pytorch_loss = pytorch_hiddens[-1].sum()
        pytorch_loss.backward()

        # NOTE: autograd backward
        # Backward pass is done differenty from HW3P1 due to mytorch interface diffrence
        # sum_backward must be implemented! 
        user_loss = np.sum(user_hiddens[-1])
        autograd.add_operation(inputs=[hx_user], output=user_loss, 
                               gradients_to_update=[None], backward_operation=sum_backward)
        autograd.backward(1)

        # NOTE: Getting input gradients is different due to interface and implementation diffrences
        # NOTE: slice_backward must be implemented!
        dx, dx_ = autograd.gradient_buffer.get_param(inp_user), inp.grad.detach()
        
        # NOTE: Getting hidden gradient is different due to interface and implementation diffrences
        # Retreived by querying the gradient bufer with the user_hiddens[0] i.e. the initial hidden state sent to the network.
        dh, dh_ = autograd.gradient_buffer.get_param(user_hiddens[0]), pytorch_hiddens[0].grad.detach().numpy()

        # NOTE: Getting network parameter gradients is different due to interface and implementation diffrences
        dW_ih, dW_hh = user_cell.ih.dW, user_cell.hh.dW
        db_ih, db_hh = user_cell.ih.db, user_cell.hh.db
        dW_ih_, dW_hh_ = pytorch_rnn_cell.weight_ih.grad.numpy(), pytorch_rnn_cell.weight_hh.grad.numpy()
        db_ih_, db_hh_ = pytorch_rnn_cell.bias_ih.grad.numpy(), pytorch_rnn_cell.bias_hh.grad.numpy()

        # Verify derivatives
        if not np.allclose(dx, dx_, rtol=1e-04):
            print('wrong value for dx in rnn cell backward')
            print(f'expected value:\n{dx_}\nGot instead:\n{dx}\n')
            return False
        if not np.allclose(dh, dh_, rtol=1e-04):
            print('wrong value for dh in rnn cell backward')
            print(f'expected value:\n{dh_}\nGot instead:\n{dh}\n')
            return False
        if not np.allclose(dW_ih, dW_ih_, rtol=1e-04):
            print('wrong value for dW_ih in rnn cell backward')
            print(f'expected value:\n{dW_ih_}\nGot instead:\n{dW_ih}\n')
            return False
        if not np.allclose(dW_hh, dW_hh_, rtol=1e-04):
            print('wrong value for dW_hh in rnn cell backward')
            print(f'expected value:\n{dW_hh_}\nGot instead:\n{dW_hh}\n')
            return False
        if not np.allclose(db_ih, db_ih_, rtol=1e-04):
            print('wrong value for db_ih in rnn cell backward')
            print(f'expected value:\n{db_ih_}\nGot instead:\n{db_ih}\n')
            return False
        if not np.allclose(db_hh, db_hh_, rtol=1e-04):
            print('wrong value for db_hh in rnn cell backward')
            print(f'expected value:\n{db_hh_}\nGot instead:\n{db_hh}\n')
            return False
        
        # Use to save test data for next semester
        # results = [dx1, dh1, dx2, dh2, dW_ih, dW_hh, db_ih, db_hh]
        # np.save(os.path.join('autograder', 'hw3_autograder',
        #                      'data', 'rnncell_backward.npy'), results, allow_pickle=True)
        return True
    

    def test_rnn_classifier(self):
        rnn_layers = 2
        batch_size = 1
        seq_len = 2
        input_size = 2
        hidden_size = 3  # hidden_size > 100 will cause precision error
        output_size = 8

        np.random.seed(11785)
        torch.manual_seed(11785)

        data_x = np.array([[[-0.5174464, -0.72699493], [0.13379902, 0.7873791],
                            [0.2546231, 0.5622532]]])
        data_y = np.random.randint(0, output_size, batch_size)

        # Initialize
        # Reference model
        rnn_model = ReferenceModel(input_size,
                                   hidden_size,
                                   output_size,
                                   rnn_layers=rnn_layers)
        model_state_dict = rnn_model.state_dict()
        
        # My model
        autograd = autograd_engine.Autograd()
        my_rnn_model = RNNPhonemeClassifier(input_size,
                                            hidden_size,
                                            output_size,
                                            autograd,
                                            num_layers=rnn_layers)
        rnn_weights = [[
            model_state_dict["rnn.weight_ih_l%d" % l].numpy(),
            model_state_dict["rnn.weight_hh_l%d" % l].numpy(),
            model_state_dict["rnn.bias_ih_l%d" % l].numpy(),
            model_state_dict["rnn.bias_hh_l%d" % l].numpy(),
        ] for l in range(rnn_layers)]
        fc_weights = [
            model_state_dict["output.weight"].numpy(),
            model_state_dict["output.bias"].numpy(),
        ]
        my_rnn_model.init_weights(rnn_weights, fc_weights)

        # Test forward pass
        # Reference model
        ref_init_h = nn.Parameter(
            torch.zeros(rnn_layers, batch_size, hidden_size, dtype=torch.float),
            requires_grad=True,
        )
        ref_out_tensor = rnn_model(torch.FloatTensor(data_x), ref_init_h)
        ref_out = ref_out_tensor.detach().numpy()

        # My model
        my_out = my_rnn_model(data_x)

        # Verify forward outputs
        print("Testing RNN Classifier Toy Example Forward...")
        if not np.allclose(my_out, ref_out, rtol=1e-03):
            print('wrong value in rnn classifier forward')
            print(f'Expected value:\n{ref_out}\nGot instead:\n{my_out}\n')
            return False
        
        # if not self.assertions(my_out, ref_out, 'closeness', 'RNN Classifier Forwrd'): #rtol=1e-03)
        # return 'RNN Forward'
        print("RNN Classifier Toy Example Forward: PASS")
        print("Testing RNN Classifier Toy Example Backward...")

        # Test backward pass
        # Reference model
        criterion = nn.CrossEntropyLoss()
        loss = criterion(ref_out_tensor, torch.LongTensor(data_y))
        ref_loss = loss.detach().item()
        rnn_model.zero_grad()
        loss.backward()
        grad_dict = {
            k: v.grad
            for k, v in zip(rnn_model.state_dict(), rnn_model.parameters())
        }
        dh = ref_init_h.grad

        # my model
        # NOTE: autograd backward
        # Backward pass is done differenty from HW3P1 due to mytorch interface diffrence
        my_criterion = SoftmaxCrossEntropy(autograd)
        my_labels_onehot = np.zeros((batch_size, output_size))
        my_labels_onehot[np.arange(batch_size), data_y] = 1.0
        my_loss = my_criterion(my_labels_onehot, my_out)
        autograd.backward(1)
        
        # NOTE: Getting gradients wrt hiddens[0] i.e the initial hidden state
        # NOTE: Gradients wrt to the hiddens at each timestep is stored in the internal Autograd gradient buffer.
        # The gradients for a particular input can be retrieved by querying the gradient buffer with the numpy.ndarray input.
        # Since, my_rnn_model.hiddens[0][t] is fed into the network, the gradient with my_rnn_model.hiddens[0][t] gets what 
        # would be my_dh[t]. Thus, this requires iterating through all timesteps to get the gradient wrt each timestep which are then 
        # concatenated for comparison with pytorch's input gradient. 
        my_dh = []
        for h in range(len(my_rnn_model.hiddens[0])):
            my_dh.append(autograd.gradient_buffer.get_param(my_rnn_model.hiddens[0][h]))
        my_dh = np.array(my_dh)

        # Verify derivative w.r.t. each network parameters
        # NOTE: Class variable names are diffrent due to implementation change
        if not np.allclose(my_dh, dh.detach().numpy(), rtol=1e-04):
            print('wrong value for dh in rnn classifier backward')
            print(f'Expected value:\n{dh.detach().numpy()}\nGot instead:\n{my_dh}\n')
            return False
        
        if not np.allclose(my_rnn_model.output_layer.dW, grad_dict["output.weight"].detach().numpy(),rtol=1e-03,):
            print('wrong value for dLdW in rnn classifier backward')
            print(f'Expected value:\n{grad_dict["output.weight"].detach().numpy()}\nGot instead:\n{my_rnn_model.output_layer.dW}\n')
            return False
        
        if not np.allclose(my_rnn_model.output_layer.db.reshape(-1,), grad_dict["output.bias"].detach().numpy()):
            print('wrong value for dLdb in rnn classifier backward')
            print(f'Expected value:\n{grad_dict["output.bias"].detach().numpy()}\nGot instead:\n{my_rnn_model.output_layer.db.reshape(-1,)}\n')
            return False
        
        for l, rnn_cell in enumerate(my_rnn_model.rnn):
            if not np.allclose(rnn_cell.ih.dW, grad_dict["rnn.weight_ih_l%d" % l].detach().numpy(), rtol=1e-03,):
                print('wrong value for dW_ih in rnn classifier backward')
                print(f'Expected value:\n{grad_dict["rnn.weight_ih_l%d" % l].detach().numpy()}\nGot instead:\n{rnn_cell.ih.dW}\n')
                return False

            if not np.allclose(rnn_cell.hh.dW, grad_dict["rnn.weight_hh_l%d" % l].detach().numpy(), rtol=1e-03,):
                print('wrong value for dW_hh in rnn classifier backward')
                print(f'Expected value:\n{grad_dict["rnn.weight_hh_l%d" % l].detach().numpy()}\nGot instead:\n{rnn_cell.hh.dW}\n')
                return False
            
            if not np.allclose(rnn_cell.ih.db, grad_dict["rnn.bias_ih_l%d" % l].detach().numpy(), rtol=1e-03,):
                print('wrong value for db_ih in rnn classifier backward')
                print(f'Expected value:\n{grad_dict["rnn.bias_ih_l%d" % l].detach().numpy()}\nGot instead:\n{rnn_cell.ih.db}\n')
                return False
            
            if not np.allclose(rnn_cell.hh.db, grad_dict["rnn.bias_hh_l%d" % l].detach().numpy(), rtol=1e-03,):
                print('wrong value for db_hh in rnn classifier backward')
                print(f'Expected value:\n{grad_dict["rnn.bias_hh_l%d" % l].detach().numpy()}\nGot instead:\n{rnn_cell.hh.db}\n')
                return False

        print("RNN Toy Classifier Backward: PASS")
        return True

    def run_test(self):
        # Toy example forward
        self.print_name("Secion 2.1 - RNN Toy Example Forward")
        toy_forward_outcome = self.test_rnncell_toy_forward()
        self.print_outcome("RNN Toy Example Forward", toy_forward_outcome)
        if toy_forward_outcome == False:
            self.print_failure("RNN Toy Example Forward")
            return False

        # Toy example backward
        self.print_name("Secion 2.2 - RNN Toy Example Backward")
        toy_backward_outcome = self.test_rnncell_toy_backward()
        self.print_outcome("RNN Toy Example Backward", toy_backward_outcome)
        if toy_backward_outcome == False:
            self.print_failure("RNN Toy Example Backward")
            return False

        # Toy example RNN Classifier
        self.print_name("Section 2.3 - RNN Classifier Toy Example")
        toy_classifier_outcome = self.test_toy_rnn_classifier()
        self.print_outcome("RNN Classifier", toy_classifier_outcome)
        if toy_classifier_outcome == False:
            self.print_failure(toy_classifier_outcome)
            return False

        return True
