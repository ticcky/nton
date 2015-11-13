import numpy as np
import random
from collections import deque, defaultdict

from nn import LSTM, OneHot, Sequential, LinearLayer, Softmax, Sigmoid, Vars
from nn.attention import Attention
from nn.switch import Switch
from db import DB
from seq_loss import SeqLoss
from data_calc import DataCalc
import metrics


class NTON(object):
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

        self.param_layers = [
            self.output_switch_p,
            self.output_rnn_clf,
            self.output_rnn,
            self.att,
            self.input_rnn,
        ]

        self.print_widths = defaultdict(dict)


    def forward(self, id_words_in, gen_n, no_print=False):
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

        #h_t, c_t = self.output_rnn.get_init()
        h_t = H[-1]
        c_t = C[-1, 0]

        ((prev_y, ), _) = self.emb.forward(([0], ))

        # print 'init prev_y', prev_y
        # print 'init h_t', h_t
        # print 'init c_t', c_t

        Y = []
        y = []
        #print '---'
        for i in range(gen_n):
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

            y_t = y_t.squeeze()

            #prev_y_ndx = np.random.choice(self.n_tokens, p=y_t)
            prev_y_ndx = y_t.argmax()
            ((prev_y, ), _) = self.emb.forward(([prev_y_ndx], ))

            Y.append(y_t)
            y.append(prev_y_ndx)

            #print self.db.vocab.rev(rnn_result_t.argmax()) #, rnn_result_t
            #print self.db.vocab.rev(db_result_t.argmax()) #, db_result_t
            #print self.db.vocab.rev(y_t.argmax()), self.db.vocab.rev(prev_y_ndx) #, db_result_t

            db_argmax = np.argmax(db_result_t)
            rnn_argmax = np.argmax(rnn_result_t)

            if not no_print:
                self.print_step('gen',
                    '  ',
                    'gen: %s' % self.db.vocab.rev(prev_y_ndx),
                    'att: %s' % query_t_aux_curr['alpha'],
                    'sw: %.2f' % p1,
                    'rnn: %s (%.2f)' % (self.db.vocab.rev(rnn_argmax), rnn_result_t[rnn_argmax]),
                    'db: %s (%.2f)' % (self.db.vocab.rev(db_argmax), db_result_t[db_argmax]),

                )

            #print self.db.vocab.rev(prev_y_ndx)

            #print 'next prev_y', prev_y_ndx, prev_y
            #print 'next h_t', h_t

        Y = np.array(Y)
        y = np.array(y)

        return ((Y, y), Vars(
            E_aux=E_aux,
            H_aux=H_aux,
            query_t_aux=query_t_aux,
            db_result_t_aux=db_result_t_aux,
            h_t_aux=h_t_aux,
            rnn_result_aux=rnn_result_aux,
            switch_p_aux=switch_p_aux,
            switch_aux=switch_aux,
            gen_n=gen_n
        ))

    def print_step(self, t, *args):
        widths = self.print_widths[t]
        for i, arg in enumerate(args):
            if not i in widths:
                widths[i] = len(arg)
            width = widths[i]
            if len(arg) > width:
                widths[i] = width = len(arg)
            print arg + " " * (width - len(arg)), ' |',

        print

    def backward(self, aux, grads):
        H_aux = aux['H_aux']
        rnn_result_aux = aux['rnn_result_aux']
        switch_p_aux = aux['switch_p_aux']
        switch_aux = aux['switch_aux']
        h_t_aux = aux['h_t_aux']
        db_result_t_aux = aux['db_result_t_aux']
        query_t_aux = aux['query_t_aux']

        dh_tm1 = None
        dc_tm1 = None
        dH = None
        for i in range(aux['gen_n'] - 1, -1, -1):
            (dp1, din1, din2, ) = Switch.backward(switch_aux[i], (grads[i], ))
            (dh_t_1, ) = self.output_switch_p.backward(switch_p_aux[i], (dp1, ))
            (dh_t_2, ) = self.output_rnn_clf.backward(rnn_result_aux[i], (din1, ))

            dh_t = dh_t_1 + dh_t_2
            if dh_tm1 is not None:
                dh_t += dh_tm1

            if dc_tm1 == None:
                dc_tm1 = np.zeros_like(dh_t)
            (dprev_y, dc_tm1, dh_tm1_1, ) = self.output_rnn.backward(h_t_aux[i], (dh_t, dc_tm1, ))

            (dquery_t, ) = self.db.backward(db_result_t_aux[i], (din2, ))

            (dH_t, dh_tm1_2, dE, ) = self.att.backward(query_t_aux[i], (dquery_t, ))

            if dH is None:
                dH = dH_t.copy()
            else:
                dH += dH_t

            dh_tm1 = (dh_tm1_1 + dh_tm1_2).squeeze()

        dC = np.zeros_like(dH)
        dC[-1] += dc_tm1.squeeze()
        dH[-1] += dh_tm1  # Output RNN back to Input RNN last state.
        dH = dH[:, np.newaxis, :]
        dC = dC[:, np.newaxis, :]
        (dE, dh0, dc0) = self.input_rnn.backward(H_aux, (dH, dC))

    def zero_grads(self):
        for layer in self.param_layers:
            layer.grads.zero()

    def update_params(self, lr):
        for layer in self.param_layers:
            layer.params.increment_by(layer.grads, factor=-lr)

    def decode(self, Y):
        res = []
        for i in range(len(Y)):
            res.append(self.db.vocab.rev(Y[i].argmax()))

        return res

    # def prepare_data(self, x):
    #     res = []
    #     for q, a in x:
    #         x_q = self.db.words_to_ids(q.split())
    #         x_a = self.db.words_to_ids(a.split())
    #         res.append((x_q, x_a))
    #
    #     return res

    def prepare_data_signle(self, (q, a)):
        x_q = self.db.words_to_ids(q)
        x_a = self.db.words_to_ids(a)

        return (x_q, x_a)



def main(**kwargs):
    eval_step = kwargs.pop('eval_step')
    np.set_printoptions(edgeitems=3,infstr='inf',
                        linewidth=200, nanstr='nan', precision=4,
                        suppress=False, threshold=1000, formatter={'float': lambda x: "%.1f" % x})
    calc = DataCalc()
    data_train = calc.gen_data(test_data=False)
    data_test = calc.gen_data(test_data=True)

    db = DB(calc.get_db(), calc.get_vocab())
    db.vocab.freeze()

    #q = db.get_vector('1+3')
    #a = db.vocab.rev(db.forward((q, ))[0][0].argmax())
    #print a


    nton = NTON(
        n_tokens=len(db.vocab),
        db=db,
        **kwargs
    )

    eval_nton(nton, data_test, 100)

    # data_train = [
    #     ("i would like chinese food", "ok chong is good"),
    #     ("what about indian", "ok taj is good"),
    #     ("give me czech", "go to hospoda"),
    #     ("i like english food", "go to tavern")
    # ]

    avg_loss = deque(maxlen=20)
    for epoch in xrange(10000000):
        x_q, x_a = nton.prepare_data_signle(next(data_train))

        nton.zero_grads()

        ((Y, y), aux) = nton.forward(x_q, len(x_a))
        ((loss, ), loss_aux) = SeqLoss.forward((Y, x_a, ))
        (dY, ) = SeqLoss.backward(loss_aux, 1.0)

        nton.backward(aux, dY)
        nton.update_params(lr=0.1)

        avg_loss.append(loss)

        #x_a_hat_str = " ".join(nton.decode(Y))
        x_a_hat_str = " ".join(db.vocab.rev(x) for x in y)
        x_a_str = " ".join(db.vocab.rev(x) for x in x_a)

        nton.print_step('loss',
                        'loss %.4f' % np.mean(avg_loss),
                        'example %d' % epoch,
                        "%s" % Y[np.arange(len(x_a)), x_a],
                        #"%s" % Y[0, [
                        #    db.vocab['0'],
                        #    db.vocab['1'],
                        #    db.vocab['2'],
                        #    db.vocab['3'],
                        #    db.vocab['4'],
                        #    db.vocab['5'],
                        #    db.vocab['6'],
                        #    db.vocab['7'],
                        #    db.vocab['8'],
                        #    db.vocab['9']
                        #]],
                        " ".join([db.vocab.rev(x) for x in x_q]), '->', x_a_hat_str,
                        "(%s)" % x_a_str,
                        "%s" % ("*" if x_a_str == x_a_hat_str else "")
        )

        if epoch % eval_step == 0:
            eval_nton(nton, 'train', data_train, 500)
            eval_nton(nton, 'test', data_test, 500)


def eval_nton(nton, data_label, data, n_examples):
    wers = []
    acc = []
    for i in xrange(n_examples):
        x_q, x_a = nton.prepare_data_signle(next(data))
        ((Y, y), aux) = nton.forward(x_q, len(x_a), no_print=True)
        wers.append(metrics.calculate_wer(x_a, y))
        acc.append(metrics.accuracy(x_a, y))

    print '### Evaluation(%s): ' % data_label,
    print '  %15.15s %.2f' % ("WER:", np.mean(wers)),
    print '  %15.15s %.2f' % ("Accuracy:", np.mean(acc)),
    print



if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--n_cells', type=int, default=16)
    parser.add_argument('--eval_step', type=int, default=1000)
    #parser.add_argument('--n_words', type=int, default=100)
    #parser.add_argument('--n_db', type=int, default=10)

    args = parser.parse_args()

    main(**vars(args))


# TODO:
#  - Saving and loading parameters.
#  - Making the task more difficult
#    - more db lookups needed per query
#    - larger db
#  - Adding Adam learning rule.
#  - Evaluation
#    - BLEU, WER, PER.
#  - Add gradient checks for NTON.
