import numpy as np

from nn import LSTM, ParametrizedBlock, Attention, Vars


class NLU(ParametrizedBlock):
    def __init__(self, n_cells, emb_dim, n_slu):
        self.input_rnn = LSTM(n_in=emb_dim, n_out=n_cells)

        self.slus = []
        for i in range(n_slu):
            self.slus.append(Attention(n_hidden=n_cells))

        self.parametrize_from_layers(
            [self.input_rnn] + self.slus,
            ["input_rnn"] + ["slu%.2d" % i for i in range(n_slu)]
        )

    def init_state(self):
        return self.input_rnn.get_init()[0]

    def forward(self, (E, )):
        h0, c0 = self.input_rnn.get_init()
        ((H, C), H_aux) = self.input_rnn.forward((E[:, np.newaxis, :], h0, c0, ))
        H = H[:, 0]
        C = C[:, 0]

        h_t = H[-1]

        slu_res = []
        slu_aux = []
        for slu in self.slus:
            ((slu_i, ), slu_i_aux) = slu.forward((H, h_t, E))
            slu_res.append(slu_i)
            slu_aux.append(slu_i_aux)

        return ((h_t, ) + tuple(slu_res), Vars(
            slu=slu_aux,
            H=H_aux
        ))

    def backward(self, aux, grads):
        assert type(grads) == tuple
        assert len(grads) == len(self.slus) + 1

        dh_t = grads[0]
        dslus = grads[1:]

        lst_dH = []
        lst_dh_t = []
        lst_dE = []

        for slu, dslu_i, slu_i_aux in zip(self.slus, dslus, aux['slu']):
            self.accum_grads(
                (lst_dH, lst_dh_t, lst_dE, ),
                slu.backward(slu_i_aux, (dslu_i, ))
            )

        dh_t = dh_t + sum(lst_dh_t)


        dH = sum(lst_dH)
        dH[-1] += dh_t

        dH = dH[:, np.newaxis, :]
        dC = np.zeros_like(dH)
        (dE, dh0, dc0) = self.input_rnn.backward(aux['H'], (dH, dC))

        dE = dE[:, 0, :] + sum(lst_dE)

        return (dE, )




        


