import numpy as np

from nn import LSTM, OneHot, Sequential, LinearLayer, Softmax, Sigmoid
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

        ((E, ), _) = self.emb.forward((id_words_in, ))
        ((H, C ), _) = self.input_rnn.forward((E[:, np.newaxis, :], h0, c0, ))
        H = H[0]

        h_t, c_t = self.output_rnn.get_init()
        ((prev_y, ), _) = self.emb.forward(([0], ))

        print 'init prev_y', prev_y
        print 'init h_t', h_t
        print 'init c_t', c_t

        for i in range(self.max_gen_steps):
            ((query_t, ), _) = self.att.forward((H, h_t, E, ))

            ((db_result_t, ), _) = self.db.forward((query_t, ))


            ((h_t, c_t), _) = self.output_rnn.forward((prev_y[:, np.newaxis, :], c_t, h_t))
            h_t = h_t[0][0]
            c_t = c_t[0][0]

            ((rnn_result_t, ), _) = self.output_rnn_clf.forward((h_t, ))
            ((p1, ), _) = self.output_switch_p.forward((h_t, ))
            ((y_t, ), _) = Switch.forward((p1, rnn_result_t, db_result_t))

            prev_y_ndx = np.random.choice(self.n_tokens, p=y_t.squeeze())
            ((prev_y, ), _) = self.emb.forward(([prev_y_ndx], ))

            print 'next prev_y', prev_y_ndx, prev_y
            print 'next h_t', h_t



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
    nton.forward("i would like chinese food".split(), "ok chong is good".split())



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cells', type=int, default=16)

    args = parser.parse_args()

    main(**vars(args))