import numpy as np


class CTC(object):

    def __init__(self, BLANK=0):
        """

        Initialize instance variables

        Argument(s)
        -----------

        BLANK (int, optional): blank label index. Default 0.

        """

        # No need to modify
        self.BLANK = BLANK

    def extend_target_with_blank(self, target):
        """Extend target sequence with blank.

        Input
        -----
        target: (np.array, dim = (target_len,))
                target output
        ex: [B,IY,IY,F]

        Return
        ------
        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended target sequence with blanks
        ex: [-,B,-,IY,-,IY,-,F,-]

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections
        ex: [0,0,0,1,0,0,0,1,0]
        """

        extended_symbols = []
        skip_connect = []

        extended_symbols.append(self.BLANK)
        skip_connect.append(0)

        for i, symbol in enumerate(target):
            extended_symbols.append(symbol)
            skip_connect.append(0)

            extended_symbols.append(self.BLANK)

            skip_connect.append(0)

            if i > 0 and target[i] != target[i - 1]:
                skip_connect[-2] = 1

        extended_symbols = np.array(extended_symbols)
        skip_connect = np.array(skip_connect)

        return extended_symbols, skip_connect

    def get_forward_probs(self, logits, extended_symbols, skip_connect):
        """Compute forward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(Symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t, qextSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        alpha = np.zeros((T, S))

        alpha[0, 0] = logits[0, self.BLANK]
        alpha[0, 1] = logits[0, extended_symbols[1]] if S > 1 else 0

        for t in range(1, T):
            for s in range(S):
                a1 = alpha[t - 1, s]  # Probability of staying in the same symbol
                a2 = alpha[t - 1, s - 1] if s > 0 else 0
                a3 = alpha[t - 1, s - 2] if s > 1 and skip_connect[s] == 1 else 0

                alpha[t, s] = (a1 + a2 + a3) * logits[t, extended_symbols[s]]

        return alpha

    def get_backward_probs(self, logits, extended_symbols, skip_connect):
        """Compute backward probabilities.

        Input
        -----
        logits: (np.array, dim = (input_len, len(symbols)))
                predict (log) probabilities

                To get a certain symbol i's logit as a certain time stamp t:
                p(t,s(i)) = logits[t,extSymbols[i]]

        extSymbols: (np.array, dim = (2 * target_len + 1,))
                    extended label sequence with blanks

        skipConnect: (np.array, dim = (2 * target_len + 1,))
                    skip connections

        Return
        ------
        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probabilities

        """

        S, T = len(extended_symbols), len(logits)
        beta = np.zeros((T, S))

        beta[T - 1, S - 1] = 1
        beta[T - 1, S - 2] = 1
        beta[T - 1, 0 : S - 3] = 0

        for t in range(T - 2, -1, -1):
            beta[t, S - 1] = beta[t + 1, S - 1] * logits[t + 1, extended_symbols[S - 1]]
            for i in range(S - 2, -1, -1):
                beta[t, i] = (
                    beta[t + 1, i] * logits[t + 1, extended_symbols[i]]
                    + beta[t + 1, i + 1] * logits[t + 1, extended_symbols[i + 1]]
                )
                if i < S - 3 and skip_connect[i + 2]:
                    beta[t, i] += (
                        beta[t + 1, i + 2] * logits[t + 1, extended_symbols[i + 2]]
                    )

        return beta

    def get_posterior_probs(self, alpha, beta):
        """Compute posterior probabilities.

        Input
        -----
        alpha: (np.array, dim = (input_len, 2 * target_len + 1))
                forward probability

        beta: (np.array, dim = (input_len, 2 * target_len + 1))
                backward probability

        Return
        ------
        gamma: (np.array, dim = (input_len, 2 * target_len + 1))
                posterior probability

        """

        T, S = alpha.shape
        gamma = np.zeros((T, S))

        for t in range(T):
            norm_factor = np.sum(alpha[t] * beta[t])
            gamma[t] = (alpha[t] * beta[t]) / norm_factor

        return gamma


class CTCLoss(object):

    def __init__(self, BLANK=0):
        """

                Initialize instance variables

        Argument(s)
                -----------
                BLANK (int, optional): blank label index. Default 0.

        """
        # -------------------------------------------->
        # No need to modify
        super(CTCLoss, self).__init__()

        self.BLANK = BLANK
        self.gammas = []
        self.ctc = CTC()

    # <---------------------------------------------

    def __call__(self, logits, target, input_lengths, target_lengths):

        # No need to modify
        return self.forward(logits, target, input_lengths, target_lengths)

    def forward(self, logits, target, input_lengths, target_lengths):
        """CTC loss forward

                Computes the CTC Loss by calculating forward, backward, and
                posterior proabilites, and then calculating the avg. loss between
                targets and predicted log probabilities

        Input
        -----
        logits [np.array, dim=(seq_length, batch_size, len(symbols)]:
                        log probabilities (output sequence) from the RNN/GRU

        target [np.array, dim=(batch_size, padded_target_len)]:
            target sequences

        input_lengths [np.array, dim=(batch_size,)]:
            lengths of the inputs

        target_lengths [np.array, dim=(batch_size,)]:
            lengths of the target

        Returns
        -------
        loss [float]:
            avg. divergence between the posterior probability and the target

        """

        # No need to modify
        self.logits = logits
        self.target = target
        self.input_lengths = input_lengths
        self.target_lengths = target_lengths

        #####  IMP:
        #####  Output losses should be the mean loss over the batch

        # No need to modify
        B, _ = target.shape
        total_loss = np.zeros(B)
        self.extended_symbols = []

        for batch_itr in range(B):
            target_seq = target[batch_itr, : target_lengths[batch_itr]]
            logit_seq = logits[: input_lengths[batch_itr], batch_itr, :]

            ext_symbols, skip_connect = self.ctc.extend_target_with_blank(target_seq)
            alpha = self.ctc.get_forward_probs(logit_seq, ext_symbols, skip_connect)
            beta = self.ctc.get_backward_probs(logit_seq, ext_symbols, skip_connect)
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            log_probs = np.log(logit_seq + 1e-10)
            total_loss[batch_itr] = -np.sum(gamma * log_probs[:, ext_symbols])

        return np.mean(total_loss)

    def backward(self):
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
        T, B, C = self.logits.shape
        dY = np.zeros_like(self.logits)

        for batch_itr in range(B):
            target_seq = self.target[batch_itr, : self.target_lengths[batch_itr]]
            logit_seq = self.logits[: self.input_lengths[batch_itr], batch_itr, :]

            ext_symbols, skip_connect = self.ctc.extend_target_with_blank(target_seq)
            alpha = self.ctc.get_forward_probs(logit_seq, ext_symbols, skip_connect)
            beta = self.ctc.get_backward_probs(logit_seq, ext_symbols, skip_connect)
            gamma = self.ctc.get_posterior_probs(alpha, beta)

            for t in range(self.input_lengths[batch_itr]):
                for s in range(len(ext_symbols)):
                    dY[t, batch_itr, ext_symbols[s]] -= gamma[t, s] / (
                        logit_seq[t, ext_symbols[s]] + 1e-10
                    )

        return dY
