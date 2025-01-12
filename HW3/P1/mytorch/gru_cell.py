import numpy as np
from mytorch.nn.activation import *


class GRUCell(object):
    """GRU Cell class."""

    def __init__(self, input_size, hidden_size):
        self.d = input_size
        self.h = hidden_size
        h = self.h
        d = self.d
        self.x_t = 0

        self.Wrx = np.random.randn(h, d)
        self.Wzx = np.random.randn(h, d)
        self.Wnx = np.random.randn(h, d)

        self.Wrh = np.random.randn(h, h)
        self.Wzh = np.random.randn(h, h)
        self.Wnh = np.random.randn(h, h)

        self.brx = np.random.randn(h)
        self.bzx = np.random.randn(h)
        self.bnx = np.random.randn(h)

        self.brh = np.random.randn(h)
        self.bzh = np.random.randn(h)
        self.bnh = np.random.randn(h)

        self.dWrx = np.zeros((h, d))
        self.dWzx = np.zeros((h, d))
        self.dWnx = np.zeros((h, d))

        self.dWrh = np.zeros((h, h))
        self.dWzh = np.zeros((h, h))
        self.dWnh = np.zeros((h, h))

        self.dbrx = np.zeros((h))
        self.dbzx = np.zeros((h))
        self.dbnx = np.zeros((h))

        self.dbrh = np.zeros((h))
        self.dbzh = np.zeros((h))
        self.dbnh = np.zeros((h))

        self.r_act = Sigmoid()
        self.z_act = Sigmoid()
        self.h_act = Tanh()

        # Define other variables to store forward results for backward here

    def init_weights(self, Wrx, Wzx, Wnx, Wrh, Wzh, Wnh, brx, bzx, bnx, brh, bzh, bnh):
        self.Wrx = Wrx
        self.Wzx = Wzx
        self.Wnx = Wnx
        self.Wrh = Wrh
        self.Wzh = Wzh
        self.Wnh = Wnh
        self.brx = brx
        self.bzx = bzx
        self.bnx = bnx
        self.brh = brh
        self.bzh = bzh
        self.bnh = bnh

    def __call__(self, x, h_prev_t):
        return self.forward(x, h_prev_t)

    def forward(self, x, h_prev_t):
        """GRU cell forward.

        Input
        -----
        x: (input_dim)
            observation at current time-step.

        h_prev_t: (hidden_dim)
            hidden-state at previous time-step.

        Returns
        -------
        h_t: (hidden_dim)
            hidden state at current time-step.

        """
        self.x = x
        self.hidden = h_prev_t

        assert self.x.shape == (self.d,)
        assert self.hidden.shape == (self.h,)

        r = (
            np.dot(self.Wrx, self.x)
            + self.brx
            + np.dot(self.Wrh, self.hidden)
            + self.brh
        )
        self.r = self.r_act.forward(r)

        z = (
            np.dot(self.Wzx, self.x)
            + self.bzx
            + np.dot(self.Wzh, self.hidden)
            + self.bzh
        )
        self.z = self.z_act.forward(z)

        n = (
            np.dot(self.Wnx, self.x)
            + self.bnx
            + self.r * (np.dot(self.Wnh, self.hidden) + self.bnh)
        )
        self.n = self.h_act.forward(n)

        assert self.r.shape == (self.h,)
        assert self.z.shape == (self.h,)
        assert self.n.shape == (self.h,)

        h_t = (1 - self.z) * self.n + self.z * self.hidden

        assert h_t.shape == (self.h,)

        return h_t

    def backward(self, delta):
        """GRU cell backward.

        This must calculate the gradients wrt the parameters and return the
        derivative wrt the inputs, xt and ht, to the cell.

        Input
        -----
        delta: (hidden_dim)
                summation of derivative wrt loss from next layer at
                the same time-step and derivative wrt loss from same layer at
                next time-step.

        Returns
        -------
        dx: (1, input_dim)
            derivative of the loss wrt the input x.

        dh_prev_t: (1, hidden_dim)
            derivative of the loss wrt the input hidden h.

        """
        dx = np.zeros((self.d,))
        dh_prev_t = np.zeros((self.h,))

        # From equation 1
        dh_prev_t += delta * self.z
        dn = delta * (1 - self.z)
        dz = delta * (self.hidden - self.n)

        # Backprop through tanh
        dn_pre_act = self.h_act.backward(dn)

        # From equation 2
        self.dWnx = np.outer(dn_pre_act, self.x)
        self.dbnx = dn_pre_act

        d_inner = dn_pre_act * self.r
        self.dWnh = np.outer(d_inner, self.hidden)
        self.dbnh = d_inner

        dr_from_n = dn_pre_act * (np.dot(self.Wnh, self.hidden) + self.bnh)
        dh_prev_t += np.dot(self.Wnh.T, d_inner)

        # Backprop through sigmoid
        dz_pre_act = self.z_act.backward(dz)

        # From equation 3
        self.dWzx = np.outer(dz_pre_act, self.x)
        self.dbzx = dz_pre_act
        self.dWzh = np.outer(dz_pre_act, self.hidden)
        self.dbzh = dz_pre_act

        dx += np.dot(self.Wzx.T, dz_pre_act)
        dh_prev_t += np.dot(self.Wzh.T, dz_pre_act)

        # Backprop through sigmoid
        dr = dr_from_n
        dr_pre_act = self.r_act.backward(dr)

        # From equation 4
        self.dWrx = np.outer(dr_pre_act, self.x)
        self.dbrx = dr_pre_act
        self.dWrh = np.outer(dr_pre_act, self.hidden)
        self.dbrh = dr_pre_act

        dx += np.dot(self.Wrx.T, dr_pre_act)
        dh_prev_t += np.dot(self.Wrh.T, dr_pre_act)

        dx += np.dot(self.Wnx.T, dn_pre_act)

        assert dx.shape == (self.d,)
        assert dh_prev_t.shape == (self.h,)

        return dx, dh_prev_t
