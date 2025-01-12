import sys, pdb, os
import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from test import Test

sys.path.append("./")

from mytorch.rnn_cell import *
from mytorch.nn.loss import *

sys.path.append("models")
from rnn_classifier import *


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
        W_ih, W_hh = (
            state_dict["weight_ih"].numpy(),
            state_dict["weight_hh"].numpy(),
        )
        b_ih, b_hh = state_dict["bias_ih"].numpy(
        ), state_dict["bias_hh"].numpy()

        # Set user cell and weights
        user_cell = RNNCell(i * 2, i * 3)
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
            print('input: \n{} \nhidden: \n{}'.format(inp[t].numpy(),
                                                      hx.detach().numpy()))
            hx = pytorch_rnn_cell(inp[t], hx)
            hx_user = user_cell(inp_user[t], hx_user)
            assert np.allclose(hx.detach().numpy(), hx_user, rtol=1e-03), \
                'wrong value for h_prime in rnn cell forward' \
                'expected: \n{} \ngot instead: \n{}'.format(hx.detach().numpy(), hx_user)
            print('*** passed ***')

        return True

    def test_rnncell_backward(self):
        expected_results = np.load(
            os.path.join("autograder", "data",
                         "toy_rnncell_backward.npy"),
            allow_pickle=True,
        )
        dx1_, dh1_, dx2_, dh2_, dW_ih_, dW_hh_, db_ih_, db_hh_ = expected_results

        np.random.seed(11785)
        torch.manual_seed(11785)

        batch_size = 1
        input_size = 2
        hidden_size = 3
        user_cell = RNNCell(2, 3)

        # Run backward once
        delta = np.random.randn(batch_size, hidden_size)
        h = np.random.randn(batch_size, hidden_size)
        h_prev_l = np.random.randn(batch_size, input_size)
        h_prev_t = np.random.randn(batch_size, hidden_size)
        dx1, dh1 = user_cell.backward(delta, h, h_prev_l, h_prev_t)

        # Run backward again
        delta = np.random.randn(batch_size, hidden_size)
        h = np.random.randn(batch_size, hidden_size)
        h_prev_l = np.random.randn(batch_size, input_size)
        h_prev_t = np.random.randn(batch_size, hidden_size)
        dx2, dh2 = user_cell.backward(delta, h, h_prev_l, h_prev_t)

        dW_ih, dW_hh = user_cell.dW_ih, user_cell.dW_hh
        db_ih, db_hh = user_cell.db_ih, user_cell.db_hh

        # Verify derivatives
        assert np.allclose(dx1, dx1_, rtol=1e-04), 'wrong value for dx in rnn cell backward (first), ' \
                                                   'expected value:\n{}\nGot instead:\n{}\n'.format(dx1_, dx1)
        assert np.allclose(dx2, dx2_, rtol=1e-04), 'wrong value for dx in rnn cell backward (second)' \
                                                   'expected value:\n{}\nGot instead:\n{}\n'.format(dx2_, dx2)
        assert np.allclose(dh1, dh1_, rtol=1e-04), 'wrong value for dh in rnn cell backward (first)' \
                                                   'expected value:\n{}\nGot instead:\n{}\n'.format(dh1_, dh1)
        assert np.allclose(dh2, dh2_, rtol=1e-04), 'wrong value for dh in rnn cell backward (second)' \
                                                   'expected value:\n{}\nGot instead:\n{}\n'.format(dh2_, dh2)
        assert np.allclose(dW_ih, dW_ih_, rtol=1e-04), 'wrong value for dW_ih in rnn cell backward' \
                                                       'expected value:\n{}\nGot instead:\n{}\n'.format(dW_ih_, dW_ih)
        assert np.allclose(dW_hh, dW_hh_, rtol=1e-04), 'wrong value for dW_hh in rnn cell backward' \
                                                       'expected value:\n{}\nGot instead:\n{}\n'.format(dW_hh_, dW_hh)
        assert np.allclose(db_ih, db_ih_, rtol=1e-04), 'wrong value for db_ih in rnn cell backward' \
                                                       'expected value:\n{}\nGot instead:\n{}\n'.format(db_ih_, db_ih)
        assert np.allclose(db_hh, db_hh_, rtol=1e-04), 'wrong value for db_hh in rnn cell backward' \
                                                       'expected value:\n{}\nGot instead:\n{}\n'.format(db_hh_, db_hh)

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
        my_rnn_model = RNNPhonemeClassifier(input_size,
                                            hidden_size,
                                            output_size,
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
            torch.zeros(rnn_layers, batch_size, hidden_size,
                        dtype=torch.float),
            requires_grad=True,
        )
        ref_out_tensor = rnn_model(torch.FloatTensor(data_x), ref_init_h)
        ref_out = ref_out_tensor.detach().numpy()

        # My model
        my_out = my_rnn_model(data_x)

        # Verify forward outputs
        print("Testing RNN Classifier Toy Example Forward...")
        assert np.allclose(my_out, ref_out, rtol=1e-03), 'wrong value in rnn classifier toy example forward\n' \
                                                         'Expected value:\n{}\nGot instead:\n{}\n' \
            .format(ref_out, my_out)
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

        # My model
        my_criterion = SoftmaxCrossEntropy()
        my_labels_onehot = np.zeros((batch_size, output_size))
        my_labels_onehot[np.arange(batch_size), data_y] = 1.0
        my_loss = my_criterion(my_out, my_labels_onehot).mean()
        delta = my_criterion.backward()
        my_dh = my_rnn_model.backward(delta)

        # Verify derivative w.r.t. each network parameters
        assert np.allclose(my_dh, dh.detach().numpy(), rtol=1e-04), \
            'wrong value for dh in rnn classifier backward\n' \
            'Expected value:\n{}\nGot instead:\n{}'.format(dh, my_dh)
        assert np.allclose(
            my_rnn_model.output_layer.dLdW,
            grad_dict["output.weight"].detach().numpy(),
            rtol=1e-03,
        ), 'wrong value for dLdW in rnn classifier backward\n' \
           'Expected value:\n{}\nGot instead:\n{}'.format(my_rnn_model.output_layer.dLdW,
                                                          grad_dict["output.weight"].detach().numpy())
        assert np.allclose(
            my_rnn_model.output_layer.dLdb.reshape(-1,), grad_dict["output.bias"].detach().numpy()
        ), 'wrong value for dLdb in rnn classifier backward\n' \
           'Expected value:\n{}\nGot instead:\n{}'.format(my_rnn_model.output_layer.dLdb.reshape(-1,),
                                                          grad_dict["output.bias"].detach().numpy())
        for l, rnn_cell in enumerate(my_rnn_model.rnn):
            assert np.allclose(
                my_rnn_model.rnn[l].dW_ih,
                grad_dict["rnn.weight_ih_l%d" % l].detach().numpy(),
                rtol=1e-03,
            ), 'wrong value for dW_ih in rnn classifier backward\n' \
               'Expected value:\n{}\nGot instead:\n{}'.format(my_rnn_model.rnn[l].dW_ih,
                                                              grad_dict["rnn.weight_ih_l%d" % l].detach().numpy())
            assert np.allclose(
                my_rnn_model.rnn[l].dW_hh,
                grad_dict["rnn.weight_hh_l%d" % l].detach().numpy(),
                rtol=1e-03,
            ), 'wrong value for dW_hh in rnn classifier backward\n' \
               'Expected value:\n{}\nGot instead:\n{}'.format(my_rnn_model.rnn[l].dW_hh,
                                                              grad_dict["rnn.weight_hh_l%d" % l].detach().numpy())
            assert np.allclose(
                my_rnn_model.rnn[l].db_ih,
                grad_dict["rnn.bias_ih_l%d" % l].detach().numpy(),
                rtol=1e-03,
            ), 'wrong value for db_ih in rnn classifier backward\n' \
               'Expected value:\n{}\nGot instead:\n{}'.format(my_rnn_model.rnn[l].db_ih,
                                                              grad_dict["rnn.bias_ih_l%d" % l].detach().numpy())
            assert np.allclose(
                my_rnn_model.rnn[l].db_hh,
                grad_dict["rnn.bias_hh_l%d" % l].detach().numpy(),
                rtol=1e-03,
            ), 'wrong value for db_hh in rnn classifier backward\n' \
               'Expected value:\n{}\nGot instead:\n{}'.format(my_rnn_model.rnn[l].db_hh,
                                                              grad_dict["rnn.bias_hh_l%d" % l].detach().numpy())

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
