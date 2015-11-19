import random
from collections import deque, defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sbt
sbt.set()

from nn import LSTM, OneHot, Sequential, LinearLayer, Softmax, Sigmoid, Vars, ParametrizedBlock
from nn.attention import Attention
from nn.switch import Switch
from db import DB
from seq_loss import SeqLoss
from data_calc import DataCalc
import metrics


class NTON(ParametrizedBlock):
    def __init__(self, n_tokens, n_cells, db, emb, max_gen=10):
        self.n_tokens = n_tokens
        self.n_cells = n_cells
        self.max_gen = max_gen

        self.db = db
        self.emb = emb

        emb_dim = emb.size()
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

        self.param_layers, self.param_layers_names = zip(*[
            (self.output_switch_p, 'switch'),
            (self.output_rnn_clf, 'out_rnn_clf'),
            (self.output_rnn, 'out_rnn'),
            (self.att, 'att'),
            (self.input_rnn, 'in_rnn'),
        ])

        self.print_widths = defaultdict(dict)

        params = {}
        grads = {}
        for layer_name, layer in zip(self.param_layers_names, self.param_layers):
            if isinstance(layer, ParametrizedBlock):
                for param_name in layer.params:
                    key = "%s__%s" % (layer_name, param_name, )
                    params[key] = layer.params[param_name]
                    grads[key] = layer.grads[param_name]

        self.parametrize(Vars(**params), Vars(**grads))

    def forward(self, (E, dec_symbol), no_print=False):
        # Process input sequence.
        h0, c0 = self.input_rnn.get_init()
        ((H, C ), H_aux) = self.input_rnn.forward((E[:, np.newaxis, :], h0, c0, ))
        H = H[:, 0]
        C = C[:, 0]

        h_tm1 = H[-1]       # Initial state of the output RNN is equal to the input RNN.
        c_tm1 = C[-1]

        # Prepare initial input symbol for generating.
        y_tm1 = dec_symbol

        Y = []
        y = []
        gen_aux = []
        for i in range(self.max_gen):   # Generate maximum `max_gen` words.
            ((y_tm1, h_tm1, c_tm1), aux_t) = self.forward_gen_step((y_tm1, h_tm1, c_tm1, H, E))

            Y.append(y_tm1.squeeze())
            gen_aux.append(aux_t)

            #prev_y_ndx = np.random.choice(self.n_tokens, p=y_t)
            y_decoded_token = y_tm1.argmax()
            y.append(y_decoded_token)

        Y = np.array(Y)
        y = np.array(y)

        return ((Y, y), Vars(
            H_aux=H_aux,
            gen_n=len(y),
            gen_aux=gen_aux
        ))

    def forward_gen_step(self, (y_tm1, h_tm1, c_tm1, H, E)):
        # print '-'
        # print 'in', y_tm1.shape
        # print 'in', h_tm1.shape
        # print 'in', c_tm1.shape
        # print 'in', H.shape
        # print 'in', E.shape


        ((h_t, c_t), h_t_aux_curr) = self.output_rnn.forward((y_tm1[np.newaxis, np.newaxis, :], h_tm1, c_tm1))
        h_t = h_t[0][0]
        c_t = c_t[0][0]

        ((rnn_result_t, ), rnn_result_aux_curr) = self.output_rnn_clf.forward((h_t, ))  # Get RNN LM result.

        ((query_t, ), query_t_aux_curr) = self.att.forward((H, h_t, E, ))      # Get the result from database.
        ((db_result_t, ), db_result_t_aux_curr) = self.db.forward((query_t, ))

        ((p1, ), switch_p_aux_curr) = self.output_switch_p.forward((h_t, ))    # Get the value of switch between RNN and database.
        ((y_t, ), aux_y_t) = Switch.forward((p1, rnn_result_t, db_result_t))   # Get switched output.
        y_t = y_t.squeeze()

        aux = Vars(
            h_t=h_t_aux_curr,
            rnn_result_t=rnn_result_aux_curr,
            query_t=query_t_aux_curr,
            db_result_t=db_result_t_aux_curr,
            p1=switch_p_aux_curr,
            y_t=aux_y_t,
        )

        self.forward_gen_step_debug(**locals())

        # print 'out', y_t.shape
        # print 'out', h_t.shape
        # print 'out', c_t.shape

        return ((y_t, h_t, c_t), aux)

    def backward_gen_step(self, aux, (dy_t, dh_t, dc_t)):
        # print 'd', dy_t.shape
        # print 'd', dh_t.shape
        # print 'd', dc_t.shape

        (dp1, drnn_result_t, ddb_result_t, ) =      Switch.backward(aux['y_t'], (dy_t , ))
        (dh_t_1, ) = self.output_switch_p.backward(aux['p1'], (dp1, ))
        (dh_t_2, ) =  self.output_rnn_clf.backward(aux['rnn_result_t'], (drnn_result_t, ))
        (dquery_t, )           =  self.db.backward(aux['db_result_t'], (ddb_result_t, ))
        (dH_t, dh_t_3, dE_t, ) = self.att.backward(aux['query_t'], (dquery_t, ))

        (dx_t, dh_tm1, dc_tm1, ) = self.output_rnn.backward(aux['h_t'], ((dh_t + dh_t_1 + dh_t_2 + dh_t_3)[None, None, :], dc_t[None, None, :], ))

        # print 'dout', dx_t[:, 0].shape
        # print 'dout', dh_tm1[0].shape
        # print 'dout', dc_tm1[0].shape
        # print 'dout', dH_t.shape
        # print 'dout', dE_t.shape

        return (dx_t[0, 0], dh_tm1[0], dc_tm1[0], dH_t, dE_t, )


    def forward_gen_step_debug(self_, y_t, db_result_t, rnn_result_t, query_t_aux_curr, p1, **kwargs):
        self = self_
        # Debug print something.
        db_argmax = np.argmax(db_result_t)
        rnn_argmax = np.argmax(rnn_result_t)
        y_t_argmax = y_t.argmax()

        self.print_step('gen',
            '  ',
            'gen: %s' % self.db.vocab.rev(y_t_argmax),
            'att: %s' % query_t_aux_curr['alpha'],
            'sw: %.2f' % p1,
            'rnn: %s (%.2f)' % (self.db.vocab.rev(rnn_argmax), rnn_result_t[rnn_argmax]),
            'db: %s (%.2f)' % (self.db.vocab.rev(db_argmax), db_result_t[db_argmax]),

        )

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

    def backward(self, aux, (grads, _)):
        H_aux = aux['H_aux']
        gen_aux = aux['gen_aux']

        dh_tp1, dc_tp1 = self.output_rnn.get_init_grad()
        dx_tp1 = np.zeros_like(grads[0])
        dH = None
        dE = None
        for i in reversed(range(aux['gen_n'])):
            (dx_tp1, dh_tp1, dc_tp1, dH_t, dE_t) = self.backward_gen_step(gen_aux[i], (dx_tp1 + grads[i], dh_tp1, dc_tp1))

            if dH is None:
                dH = dH_t.copy()
            else:
                dH += dH_t

            if dE is None:
                dE = dE_t.copy()
            else:
                dE += dE_t


        dH[-1] += dh_tp1.squeeze()  # Output RNN back to Input RNN last state.
        dC = np.zeros_like(dH)
        dC[-1] += dc_tp1.squeeze()
        dH = dH[:, np.newaxis, :]
        dC = dC[:, np.newaxis, :]
        (dE_2, dh0, dc0) = self.input_rnn.backward(H_aux, (dH, dC))

        dE += dE_2[:, 0, :]

        return (dE, dx_tp1)

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


def plot(losses, plot_filename):
    """Plot learning curve."""
    pal = sbt.color_palette()
    col1 = pal[0]
    col2 = pal[2]
    fig, ax1 = plt.subplots()

    ax1.plot(losses, '-', linewidth=1, color=col1)
    ax1.set_xlabel('Example')
    ax1.set_ylabel('Avg loss', color=col1)
    #ax1.set_ylim([0.0, 1.0])

    #ax2 = ax1.twinx()
    #ax2.plot(m_ppx, '-o', linewidth=1, color=col2)
    #ax2.set_ylabel('Perplexity', color=col2)

    plt.savefig(plot_filename)


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
    emb = OneHot(n_tokens=len(db.vocab))

    nton = NTON(
        n_tokens=len(db.vocab),
        db=db,
        emb=emb,
        **kwargs
    )

    eval_nton(nton, emb, db, 'prep_test', data_test, 1)

    # data_train = [
    #     ("i would like chinese food", "ok chong is good"),
    #     ("what about indian", "ok taj is good"),
    #     ("give me czech", "go to hospoda"),
    #     ("i like english food", "go to tavern")
    # ]

    avg_loss = deque(maxlen=20)
    losses = []
    for epoch in xrange(10000000):
        x_q, x_a = nton.prepare_data_signle(next(data_train))

        nton.zero_grads()

        # Prepare input.
        ((x_q_emb, ), _) = emb.forward((x_q, ))
        ((symbol_dec, ), _) = emb.forward(([db.vocab['[EOS]']], ))
        symbol_dec = symbol_dec[0]

        ((Y, y), aux) = nton.forward((x_q_emb, symbol_dec))
        ((loss, ), loss_aux) = SeqLoss.forward((Y, x_a, ))
        (dY, ) = SeqLoss.backward(loss_aux, 1.0)

        nton.backward(aux, (dY, None ))
        nton.update_params(lr=0.1)

        avg_loss.append(loss)

        #x_a_hat_str = " ".join(nton.decode(Y))
        x_a_hat_str = " ".join(db.vocab.rev(x) for x in y)
        x_a_str = " ".join(db.vocab.rev(x) for x in x_a)

        mean_loss = np.mean(avg_loss)
        losses.append(mean_loss)

        nton.print_step('loss',
                        'loss %.4f' % mean_loss,
                        'example %d' % epoch,
                        "%s" % Y[np.arange(min(len(x_a), len(Y))), x_a[:min(len(x_a), len(Y))]],
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
        print

        if epoch % eval_step == 0 and epoch > 0:
            eval_nton(nton, emb, db, 'train', data_train, 200)
            eval_nton(nton, emb, db, 'test', data_test, 200)

        if epoch % 100 == 0:
            plot(losses, 'lcurve.png')


def eval_nton(nton, emb, db, data_label, data, n_examples):
    print '### Evaluation(%s): ' % data_label
    wers = []
    acc = []
    for i in xrange(n_examples):
        x_q, x_a = nton.prepare_data_signle(next(data))
        ((x_q_emb, ), _) = emb.forward((x_q, ))
        ((symbol_dec, ), _) = emb.forward(([db.vocab['[EOS]']], ))
        symbol_dec = symbol_dec[0]
        print "Q:", " ".join([db.vocab.rev(x) for x in x_q])
        print "A:", " ".join([db.vocab.rev(x) for x in x_a])
        ((Y, y), aux) = nton.forward((x_q_emb, symbol_dec))

        if 0 in y:
            y = y[:np.where(y == 0)[0][0]]

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
    parser.add_argument('--n_cells', type=int, default=50)
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
