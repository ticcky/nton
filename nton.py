import random
from collections import deque, defaultdict
import time

import numpy as np
from numpy.linalg import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sbt
sbt.set()

from nn import (LSTM, OneHot, Sequential, LinearLayer, Softmax, Sigmoid, Vars,
                ParametrizedBlock, VanillaSGD, Adam, Embeddings, LearnableInput,
                ReLU)
from nn.attention import Attention
from nn.switch import Switch
from dbn import DBN
from seq_loss import SeqLoss
from data_caminfo import DataCamInfo
import metrics

from util import hplot


class NTON(ParametrizedBlock):
    def __init__(self, n_tokens, n_cells, dbs, emb, vocab, max_gen=10):
        self.n_tokens = n_tokens
        self.n_cells = n_cells
        self.max_gen = max_gen
        self.vocab = vocab

        self.dbs_keys, self.dbs = zip(*dbs.iteritems())

        self.n_dbs = len(dbs)
        self.db_key_len = 3
        for db in dbs.values():
            assert db.n == self.db_key_len

        self.emb = emb

        n_ndx = 5

        self.ndx_emb = LearnableInput((n_ndx, n_cells))
        #self.ndx_x = np.array([DBN.get_1hot_from(DBN.item_pattern % i, vocab) for i in range(n_ndx)])

        emb_dim = emb.size()
        self.input_rnn = LSTM(n_in=emb_dim, n_out=n_cells)
        self.output_rnn = LSTM(n_in=emb_dim, n_out=n_cells)
        self.output_rnn_clf = Sequential([
            LinearLayer(n_in=n_cells, n_out=n_tokens),
            Softmax()
        ])

        self.att_switch = Sequential([
            #LinearLayer(n_in=n_cells, n_out=n_cells),
            #ReLU(),
            LinearLayer(n_in=n_cells, n_out=8),
            Softmax()
        ])

        self.atts = []
        for i in range(self.db_key_len):
            self.atts.append(Attention(n_hidden=n_cells))

        self.att_ndx = Attention(n_hidden=n_cells)

        self.param_layers, self.param_layers_names = zip(*(
            [
                (self.output_rnn_clf, 'out_rnn_clf'),
                (self.output_rnn, 'out_rnn'),
                (self.att_switch, 'att_switch'),
                (self.att_ndx, 'att_ndx'),
                (self.input_rnn, 'in_rnn'),
            ] + zip(
                self.atts,
                ('att%d' % i for i in range(len(self.atts)))
            )
        ))

        self.print_widths = defaultdict(dict)

        self.parametrize_from_layers(self.param_layers, self.param_layers_names)

    def forward(self, (E, eos_token, x_a_emb), no_print=False):
        t = time.time()
        h0, c0 = self.input_rnn.get_init()
        ((H, C ), H_aux) = self.input_rnn.forward((E[:, np.newaxis, :], h0, c0, ))   # Process input sequence.
        H = H[:, 0]
        C = C[:, 0]

        h_tm1 = H[-1]       # Initial state of the output RNN is equal to the input RNN.
        c_tm1 = C[-1]

        y_tm1 = eos_token   # Prepare initial input symbol for generating.

        Y = []
        y = []
        gen_aux = []
        for i in range(self.max_gen):   # Generate maximum `max_gen` words.
            t = time.time()
            ((y_tm1, h_tm1, c_tm1), aux_t) = self.forward_gen_step((y_tm1, h_tm1, c_tm1, H, E))

            Y.append(y_tm1.squeeze())
            gen_aux.append(aux_t)

            #prev_y_ndx = np.random.choice(self.n_tokens, p=y_t)

            y_decoded_token = y_tm1.argmax()
            y.append(y_decoded_token)

            if x_a_emb != None:
                y_tm1 = x_a_emb[i - 1]
            else:
                y_tm1 = np.zeros_like(eos_token)
                y_tm1[y_decoded_token] = 1

        Y = np.array(Y)
        y = np.array(y)

        return ((Y, y), Vars(
            H_aux=H_aux,
            gen_n=len(y),
            gen_aux=gen_aux
        ))

    def forward_gen_step(self, (y_tm1, h_tm1, c_tm1, H, E)):
        ((h_t, c_t), h_t_aux_curr) = self.output_rnn.forward((y_tm1[np.newaxis, np.newaxis, :], h_tm1, c_tm1))
        h_t = h_t[0][0]
        c_t = c_t[0][0]

        ((rnn_result_t, ), rnn_result_aux_curr) = self.output_rnn_clf.forward((h_t, ))  # Get RNN LM result.

        queries_t = []
        queries_t_aux = []
        for att in self.atts:
            ((query_a_t, ), query_a_t_aux_curr) = att.forward((H, h_t, E, ))      # Get the result from database.
            queries_t.append(query_a_t)
            queries_t_aux.append(query_a_t_aux_curr)

        ((ndx_emb, ), ndx_emb_aux) = self.ndx_emb.forward(())
        #((query_ndx_t, ), query_ndx_t_aux) = self.att_ndx.forward((ndx_emb, h_t, self.ndx_x))

        total_query = tuple(queries_t) #+ (np.ones_like(query_ndx_t), )  # !!! TODO

        db_results_t = []
        db_results_t_aux = []
        for db in self.dbs:
            ((db_i_result_t, ), db_i_result_t_aux_curr) = db.forward(total_query)
            db_results_t.append(db_i_result_t)
            db_results_t_aux.append(db_i_result_t_aux_curr)

        #att_sw_in_t = np.zeros((len(db_results_t) + 1, self.n_cells))
        #att_sw_out_t = np.array([rnn_result_t] + db_results_t)
        #((y_t, ), y_t_aux) = self.att_switch.forward((att_sw_in_t, h_t, att_sw_out_t))
        ((swp_t, ), swp_t_aux) = self.att_switch.forward((h_t, ))

        ((y_t, ), y_t_aux) = Switch.forward((swp_t, rnn_result_t, ) + tuple(db_results_t))

        aux = Vars(
            h_t=h_t_aux_curr,
            rnn_result_t=rnn_result_aux_curr,
            queries_t=queries_t_aux,
            ndx_emb=ndx_emb_aux,
            #query_ndx_t=query_ndx_t_aux,
            db_results_t=db_results_t_aux,
            swp_t=swp_t_aux,
            y_t=y_t_aux,
        )

        self.forward_gen_step_debug(**locals())

        return ((y_t, h_t, c_t), aux)

    def backward_gen_step(self, aux, (dy_t, dh_t, dc_t)):
        dh_t_lst = [dh_t]

        dswitch = Switch.backward(aux['y_t'], (dy_t, ))
        dswp_t = dswitch[0]
        drnn_result_t = dswitch[1]
        ddb_results_t = dswitch[2:]

        (dh_t, ) = self.att_switch.backward(aux['swp_t'], (dswp_t, ))
        dh_t_lst.append(dh_t)

        dtotal_query = None

        #print
        #print 'dymax', norm(dy_t)
        for db, ddb, db_aux in zip(self.dbs, ddb_results_t, aux['db_results_t']):
            dtotal_query_i = db.backward(db_aux, (ddb, ))
            #print 'ddbmax', norm(ddb)
            #print [norm(x) for x in dtotal_query_i]

            if dtotal_query == None:
                dtotal_query = tuple(np.zeros_like(x) for x in dtotal_query_i)

            for dtotal_query_ij, dtotal_query_j in zip(dtotal_query_i, dtotal_query):
                dtotal_query_j += dtotal_query_ij

        dqueries_t = dtotal_query #[:-1]
        #dquery_ndx_t = dtotal_query[-1]

        #(dndx_emb, dh_t, _, ) = self.att_ndx.backward(aux['query_ndx_t'], (dquery_ndx_t, ))
        #dh_t_lst.append(dh_t)

        #self.ndx_emb.backward(aux['ndx_emb'], (dndx_emb, ))

        t = time.time()
        dE_lst = []
        dH_lst = []
        for att, att_aux, dquery_t in zip(self.atts, aux['queries_t'], dqueries_t):
            (dH, dh_t, dE, ) = att.backward(att_aux, (dquery_t, ))
            dH_lst.append(dH)
            dh_t_lst.append(dh_t)
            dE_lst.append(dE)

        dE_t = sum(dE_lst)
        dH_t = sum(dH_lst)

        t = time.time()
        (dh_t, ) = self.output_rnn_clf.backward(aux['rnn_result_t'], (drnn_result_t, ))
        dh_t_lst.append(dh_t)

        dh_t = sum(dh_t_lst)

        (dx_t, dh_tm1, dc_tm1, ) = self.output_rnn.backward(aux['h_t'], ((dh_t)[None, None, :], dc_t[None, None, :], ))

        return (dx_t[0, 0], dh_tm1[0], dc_tm1[0], dH_t, dE_t, )


    def forward_gen_step_debug(self_, y_t, db_results_t, rnn_result_t, queries_t_aux, swp_t_aux, **kwargs):
        self = self_
        # Debug print something.
        db_argmax = np.argmax(db_results_t, axis=1)
        rnn_argmax = np.argmax(rnn_result_t)
        y_t_argmax = y_t.argmax()

        atts = []
        for qaux, field in zip(queries_t_aux, DataCamInfo.query_fields):
            atts.append("att_%s: %s" % (field, hplot(qaux['alpha'])))

        db_results = []
        for db_key, db_res in zip(self.dbs_keys, db_results_t):
            db_argmax = db_res.argmax()
            db_results.append(
                '%s[%s, %.2f]' % (db_key, self.vocab.rev(db_argmax), db_res[db_argmax])
            )

        self.print_step('gen',
            '     #> ',
            'gen: %s' % self.vocab.rev(y_t_argmax),
            " ".join(atts))
        self.print_step('gen2',
            '     # ',
            #'att_ndx: %s' % hplot(query_ndx_t_aux['alpha']),
            'att_sw: %s' % hplot(swp_t_aux['yaux'][-1]['y'], 1.0),
            'rnn: %s (%.2f)' % (self.vocab.rev(rnn_argmax), rnn_result_t[rnn_argmax]))
        self.print_step('gen3',
            '     # ',
            " ".join(db_results),
        )

    def print_step(self, t, *args):
        widths = self.print_widths[t]
        for i, arg in enumerate(args):
            if not i in widths:
                widths[i] = len(arg)
            width = widths[i]
            if len(arg) > width:
                widths[i] = width = len(arg)
            #print arg + " " * (width - len(arg)), ' |',
            print arg,

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
        self.params.increment_by(self.grads, factor=-lr)
        # for layer in self.param_layers:
        #     layer.params.increment_by(layer.grads, factor=-lr)

    def update_params_adam(self, lr):
        for layer in self.param_layers:
            layer.params.increment_by(layer.grads, factor=-lr)

    def decode(self, Y):
        res = []
        for i in range(len(Y)):
            res.append(self.db.vocab.rev(Y[i].argmax()))

        return res

    def prepare_data_signle(self, (q, a)):
        x_q = self.vocab.words_to_ids(q)
        x_a = self.vocab.words_to_ids(a)

        return (x_q, x_a)


def plot(losses, eval_index, (train_wers, train_accs), (test_wers, test_accs), plot_filename):
    """Plot learning curve."""
    pal = sbt.color_palette()
    col1 = pal[0]
    col2 = pal[2]
    fig, ax1 = plt.subplots(linewidth=1)

    ax2 = ax1.twinx()

    plots = []
    plot_labels = []
    plots.append(ax1.plot(np.array(losses), '-', linewidth=1, color=pal[0])[0])
    plot_labels.append('Perplexity')
    #ax1.set_xlabel('Example')
    #ax1.set_ylabel('Avg loss', color=col1)
    #ax1.legend()



    #plots.append(ax2.plot(eval_index, train_wers, 'o-', label='Train WER', markersize=2, linewidth=1)[0])
    #plot_labels.append('Train WER')
    #plots.append(ax2.plot(eval_index, train_accs, 'o-', label='Train Acc', markersize=2, linewidth=1)[0])
    #plot_labels.append('Train Acc')
    plots.append(ax2.plot(eval_index, test_wers, 'o-', label='Test WER', markersize=2, linewidth=1, color=pal[1])[0])
    plot_labels.append('Test WER')
    plots.append(ax2.plot(eval_index, test_accs, 'o-', label='Test Acc', markersize=2, linewidth=1, color=pal[2])[0])
    plot_labels.append('Test Acc')

    #ax2.legend()

    fig.legend(plots, plot_labels, 'upper right')
    #legend = plt.legend(loc='upper right')

    #ax1.set_ylim([0.0, 1.0])

    #ax2 = ax1.twinx()
    #ax2.plot(m_ppx, '-o', linewidth=1, color=col2)
    #ax2.set_ylabel('Perplexity', color=col2)

    plt.savefig(plot_filename)
    plt.close()


def main(**kwargs):
    eval_step = kwargs.pop('eval_step')
    # np.set_printoptions(edgeitems=3,infstr='inf',
    #                     linewidth=200, nanstr='nan', precision=4,
    #                     suppress=False, threshold=1000, formatter={'float': lambda x: "%.1f" % x})
    cam_info = DataCamInfo()
    data_train = cam_info.gen_data(test_data=False)
    data_test = cam_info.gen_data(test_data=True)

    dbs = {}
    for field in cam_info.fields:
        db = DBN(cam_info.get_db_for(cam_info.query_fields, field), cam_info.get_vocab())
        db.vocab.freeze()

        dbs[field] = db

    assert len(dbs['food'].vocab) == len(dbs['area'].vocab)
    vocab = dbs['food'].vocab
    emb = OneHot(n_tokens=len(vocab))

    nton = NTON(
        n_tokens=len(vocab),
        dbs=dbs,
        emb=emb,
        vocab=vocab,
        **kwargs
    )

    update_rule = Adam(nton.params, nton.grads)

    eval_nton(nton, emb, vocab, 'prep_test', data_test, 1)

    # data_train = [
    #     ("i would like chinese food", "ok chong is good"),
    #     ("what about indian", "ok taj is good"),
    #     ("give me czech", "go to hospoda"),
    #     ("i like english food", "go to tavern")
    # ]

    avg_loss = deque(maxlen=20)
    losses = []
    train_wers = []
    train_accs = []
    test_wers = []
    test_accs = []
    eval_index = []
    for epoch in xrange(10000000):
        x_q, x_a = nton.prepare_data_signle(next(data_train))

        nton.print_step("q",
                        "   Q:",
                        " ".join([db.vocab.rev(x) for x in x_q]))

        nton.zero_grads()

        # Prepare input.
        ((x_q_emb, ), _) = emb.forward((x_q, ))
        x_q_emb = np.vstack((np.ones(x_q_emb.shape[1]), x_q_emb, ))

        ((x_a_emb, ), _) = emb.forward((x_a, ))
        ((symbol_dec, ), _) = emb.forward(([db.vocab['[EOS]']], ))
        symbol_dec = symbol_dec[0]

        nton.max_gen = len(x_a)
        t = time.time()
        ((Y, y), aux) = nton.forward((x_q_emb, symbol_dec, x_a_emb))
        ((loss, ), loss_aux) = SeqLoss.forward((Y, np.array(list(x_a) + [-1] * (len(y) - len(x_a))), ))
        (dY, ) = SeqLoss.backward(loss_aux, 1.0)

        nton.backward(aux, (dY, None ))

        #nton.update_params(lr=0.1)
        update_rule.update()

        avg_loss.append(loss)

        #x_a_hat_str = " ".join(nton.decode(Y))
        x_a_hat_str = " ".join(db.vocab.rev(x) for x in y)
        x_a_str = " ".join(db.vocab.rev(x) for x in x_a)

        mean_loss = np.mean(avg_loss)
        losses.append(mean_loss)

        nton.print_step('loss',
                        'loss %.4f' % mean_loss,
                        'example %d' % epoch)
        nton.print_step("y",
                        "   Y:",
                        "%s" % hplot(Y[np.arange(min(len(x_a), len(Y))), x_a[:min(len(x_a), len(Y))]], 1.0))
        nton.print_step("a",
                        "   A:",
                        x_a_hat_str)
        nton.print_step("ea",
                        "   X:",
                        "(%s)" % x_a_str,
                        "%s" % ("*" if x_a_str == x_a_hat_str else "")
        )
        print

        if epoch % eval_step == 0 and epoch > 0:
            #train_wer, train_acc = eval_nton(nton, emb, db, 'train', data_train, 200)
            test_wer, test_acc = eval_nton(nton, emb, vocab, 'test', data_test, 30)

            #train_wers.append(train_wer)
            #train_accs.append(train_acc)
            test_wers.append(test_wer)
            test_accs.append(test_acc)
            eval_index.append(epoch)

        if epoch % 100 == 0:
            plot(losses, eval_index, (train_wers, train_accs), (test_wers, test_accs), 'lcurve.png')


def eval_nton(nton, emb, vocab, data_label, data, n_examples):
    print '### Evaluation(%s): ' % data_label
    wers = []
    acc = []
    for i in xrange(n_examples):
        x_q, x_a = nton.prepare_data_signle(next(data))

        ((x_q_emb, ), _) = emb.forward((x_q, ))
        x_q_emb = np.vstack((np.ones(x_q_emb.shape[1]), x_q_emb, ))

        ((symbol_dec, ), _) = emb.forward(([vocab['[EOS]']], ))
        symbol_dec = symbol_dec[0]
        print "Q:", " ".join([vocab.rev(x) for x in x_q])
        print "A:", " ".join([vocab.rev(x) for x in x_a])
        ((Y, y), aux) = nton.forward((x_q_emb, symbol_dec, None))

        if 0 in y:
            y = y[:np.where(y == 0)[0][0]]

        wers.append(metrics.calculate_wer(x_a, y))
        acc.append(metrics.accuracy(x_a, y))

    print '### Evaluation(%s): ' % data_label,
    print '  %15.15s %.2f' % ("WER:", np.mean(wers)),
    print '  %15.15s %.2f' % ("Accuracy:", np.mean(acc)),
    print

    return np.mean(wers), np.mean(acc)



if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    from util import pdb_on_error
    pdb_on_error()

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
#  x Adding Adam learning rule.
#  - Evaluation
#    - BLEU, WER, PER.
#  x Add gradient checks for NTON.
