import numpy as np

from nn import LSTM, OneHot, Sequential, LinearLayer, Softmax, Sigmoid, Vars
from nn.attention import Attention
from nn.switch import Switch
from db import DB


class NTON(object):
    max_gen_steps = 7

    def __init__(self, n_tokens, n_cells, db):
        self.n_tokens = n_tokens
        self.n_cells = n_cells

        self.db = db


        self.emb = OneHot(n_tokens=n_tokens)
        emb_dim = len(self.db.vocab)

        self.input_rnn = LSTM(n_in=emb_dim, n_out=n_cells)
        self.output_rnn = LSTM(n_in=emb_dim, n_out=n_cells)
        self.output_rnn_clf = Sequential([
            LinearLayer(n_in=n_cells, n_out=n_tokens),
            Softmax()
        ])
        self.output_switch_p = Sequential([
            LinearLayer(n_in=n_cells, n_out=1),
            Sigmoid()
        ])
        self.att = Attention(n_hidden=n_cells)


    def forward(self, words_in, words_out):
        id_words_in = self.db.words_to_ids(words_in)
        id_words_out = self.db.words_to_ids(words_out)

        h0, c0 = self.input_rnn.get_init()

        ((E, ), E_aux) = self.emb.forward((id_words_in, ))
        ((H, C ), H_aux) = self.input_rnn.forward((E[:, np.newaxis, :], h0, c0, ))

        H = H[:, 0]

        # Generation.
        query_t_aux = []
        db_result_t_aux = []
        h_t_aux = []
        rnn_result_aux = []
        switch_p_aux = []
        switch_aux = []

        h_t, c_t = self.output_rnn.get_init()
        ((prev_y, ), _) = self.emb.forward(([0], ))

        print 'init prev_y', prev_y
        print 'init h_t', h_t
        print 'init c_t', c_t

        for i in range(self.max_gen_steps):
            ((query_t, ), query_t_aux_curr) = self.att.forward((H, h_t, E, ))

            ((db_result_t, ), db_result_t_aux_curr) = self.db.forward((query_t, ))


            ((h_t, c_t), h_t_aux_curr) = self.output_rnn.forward((prev_y[:, np.newaxis, :], c_t, h_t))
            h_t = h_t[0][0]
            c_t = c_t[0][0]

            ((rnn_result_t, ), rnn_result_aux_curr) = self.output_rnn_clf.forward((h_t, ))
            ((p1, ), switch_p_aux_curr) = self.output_switch_p.forward((h_t, ))
            ((y_t, ), switch_aux_curr) = Switch.forward((p1, rnn_result_t, db_result_t))

            query_t_aux.append(query_t_aux_curr)
            db_result_t_aux.append(db_result_t_aux_curr)
            h_t_aux.append(h_t_aux_curr)
            rnn_result_aux.append(rnn_result_aux_curr)
            switch_p_aux.append(switch_p_aux_curr)
            switch_aux.append(switch_aux_curr)

            prev_y_ndx = np.random.choice(self.n_tokens, p=y_t.squeeze())
            ((prev_y, ), _) = self.emb.forward(([prev_y_ndx], ))

            #print 'next prev_y', prev_y_ndx, prev_y
            #print 'next h_t', h_t

        return Vars(
            E_aux=E_aux,
            H_aux=H_aux,
            query_t_aux=query_t_aux,
            db_result_t_aux=db_result_t_aux,
            h_t_aux=h_t_aux,
            rnn_result_aux=rnn_result_aux,
            switch_p_aux=switch_p_aux,
            switch_aux=switch_aux
        )

    def backward(self, aux, grads):
        H_aux = aux['H_aux']
        rnn_result_aux = aux['rnn_result_aux']
        switch_p_aux = aux['switch_p_aux']
        switch_aux = aux['switch_aux']
        h_t_aux = aux['h_t_aux']
        db_result_t_aux = aux['db_result_t_aux']
        query_t_aux = aux['query_t_aux']

        dh_tm1 = None
        dH = None
        for i in range(self.max_gen_steps - 1, -1, -1):
            (dp1, din1, din2, ) = Switch.backward(switch_aux[i], (grads[i], ))
            (dh_t_1, ) = self.output_switch_p.backward(switch_p_aux[i], (dp1, ))
            (dh_t_2, ) = self.output_rnn_clf.backward(rnn_result_aux[i], (din1, ))

            dh_t = dh_t_1 + dh_t_2
            if dh_tm1 != None:
                dh_t += dh_tm1

            dc_t = np.zeros_like(dh_t)
            (dprev_y, dc_tm1, dh_tm1_1, ) = self.output_rnn.backward(h_t_aux[i], (dh_t, dc_t, ))

            (dquery_t, ) = self.db.backward(db_result_t_aux[i], (din2, ))

            (dH, dh_tm1_2, dE, ) = self.att.backward(query_t_aux[i], (dquery_t, ))

            dh_tm1 = (dh_tm1_1 + dh_tm1_2).squeeze()

        dH = dH[:, np.newaxis, :]
        (dE, dh0, dc0) = self.input_rnn.backward(H_aux, (dH, np.zeros_like(dH)))



def main(**kwargs):
    np.set_printoptions(edgeitems=3,infstr='inf',
                        linewidth=200, nanstr='nan', precision=4,
                        suppress=False, threshold=1000, formatter=None)
    db = DB()
    db.vocab.freeze()

    nton = NTON(
        n_tokens=len(db.vocab),
        db=db,
        **kwargs
    )
    aux = nton.forward("i would like chinese food".split(), "ok chong is good".split())

    nton.backward(aux, [np.ones(nton.n_tokens)] * nton.max_gen_steps)



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cells', type=int, default=16)

    args = parser.parse_args()

    main(**vars(args))