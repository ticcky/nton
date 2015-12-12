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

from nn import (LSTM, OneHotFromVocab, Sequential, LinearLayer, Softmax, Sigmoid, Vars,
                ParametrizedBlock, VanillaSGD, Adam, Embeddings, LearnableInput,
                ReLU, Sum)
from nn.attention import Attention
from nn.switch import Switch
import modules
from dbn import DBN
from seq_loss import SeqLoss
from data_caminfo import DataCamInfo
import metrics

from util import hplot


class NTON(ParametrizedBlock):
    def __init__(self, n_cells, mgr_h_dims, db_index, db_contents, db_mapping, vocab, index_vocab):
        self.n_cells = n_cells

        self.vocab = vocab
        self.n_db_keys = len(db_contents)
        vocab_size = len(vocab)

        self.one_hot = OneHotFromVocab(vocab)
        self.mgr = modules.Manager(n_cells, n_cells, 1, mgr_h_dims)
        self.nlu = modules.NLU(n_cells, vocab_size, self.n_db_keys)
        self.nlg = modules.NLG(vocab_size, n_cells, db_mapping)
        self.tracker = modules.Tracker(n_cells, n_cells)
        self.dbset = modules.DBSet(db_index, db_contents, vocab, index_vocab)

        self.parametrize_from_layers(
            [self.mgr, self.nlu, self.nlg, self.tracker],
            ["mgr", "nlu", "nlg", "tracker"]
        )

        self.print_widths = defaultdict(dict)

    def forward(self, dialog):
        s_t = self.mgr.init_state()
        tr_tm1 = tuple(np.ones((len(self.vocab), )) for _ in range(self.n_db_keys))
        slu = tuple(np.zeros((len(self.vocab), )) for _ in range(self.n_db_keys))

        h_t = self.nlu.init_state()

        for sys, usr in dialog:
            print 'sys', sys
            print 'usr', usr
            ((O_t, ), O_t_aux) = self.one_hot.forward((("<start>", ) + tuple(sys.split()), ))
            ((I_t, ), I_t_aux) = self.one_hot.forward((usr.split(), ))

            (db_res_t, db_res_t_aux) = self.dbset.forward(tr_tm1)
            db_dist_t = db_res_t[0]
            db_t = db_res_t[1:]
            ((O_hat_t, ), O_hat_t_aux) = self.nlg.forward((s_t, O_t, 0, ) + db_t + tr_tm1 + slu)

            (nlu_t, nlu_t_aux) = self.nlu.forward((I_t, ))
            h_t = nlu_t[0]

            tr_t = []
            for tr_tm1_i, nlu_t_i in zip(tr_tm1, nlu_t[1:]):
                ((tr_t_i, ), tr_t_i_aux) = self.tracker.forward((tr_tm1_i, nlu_t_i, h_t, s_t, ))
                tr_t.append(tr_t_i)

            ((db_count_t, ), db_count_t_aux) = Sum.forward((db_dist_t, ))

            self.mgr.forward((s_t, h_t, db_count_t, ))

            tr_tm1 = tuple(tr_t)
            print 'done'


    def backward(self, aux, (grads, _)):
        pass

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
