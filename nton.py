import random
from collections import deque, defaultdict
import time

import numpy as np
from numpy.linalg import norm

import nn
from nn import (LSTM, OneHotFromVocab, Sequential, LinearLayer, Softmax, Sigmoid, Vars,
                ParametrizedBlock, VanillaSGD, Adam, Embeddings, LearnableInput,
                ReLU, Sum, OneHot)
from nn.attention import Attention
from nn.switch import Switch
import modules
from dbn import DBN
from data_caminfo import DataCamInfo
import metrics

from util import hplot
from vocab import Vocab


class NTON(ParametrizedBlock):
    def __init__(self, n_cells, mgr_h_dims, n_db_keys, db_index, db_contents, db_mapping, vocab, index_vocab):
        self.n_cells = n_cells

        self.vocab = vocab
        nn.DEBUG.vocab = vocab
        nn.DEBUG.db_mapping = db_mapping

        self.n_db_keys = n_db_keys
        vocab_size = len(vocab)

        self.one_hot = OneHotFromVocab(vocab)
        self.mgr = modules.Manager(n_cells, n_cells, 1, mgr_h_dims)
        self.nlu = modules.NLU(n_cells, vocab_size, self.n_db_keys)
        self.nlg = modules.NLG(vocab_size, n_cells, db_mapping)
        self.trackers = modules.TrackerSet(n_cells, n_cells, self.n_db_keys)
        #self.tracker = modules.Tracker(n_cells, n_cells)
        self.dbset = modules.DBSet(db_index, db_contents, vocab, index_vocab)

        extra_params = dict(
            mgr_init_state=self.mgr.get_zero_state()
        )

        extra_grads = dict(
            mgr_init_state=self.mgr.get_zero_state()
        )

        self.parametrize_from_layers(
            [self.mgr, self.nlu, self.nlg, self.tracker],
            ["mgr", "nlu", "nlg", "tracker"],
            extra_params=extra_params,
            extra_grads=extra_grads
        )

        self.print_widths = defaultdict(dict)

    def get_labels(self, dialog):
        res = []
        for sys, usr in dialog:
            res.append(map(self.vocab.get, sys.split()))

        return res

    def forward_dialog(self, dialog):
        num_dialog = []
        for sys, usr in dialog:
            ((O_t, ), _) = self.one_hot.forward((("<start>", ) + tuple(sys.split()), ))
            ((I_t, ), _) = self.one_hot.forward((usr.split(), ))

            num_dialog.append(O_t)
            num_dialog.append(I_t)

        return self.forward(num_dialog)

    def forward(self, dialog):
        """Takes dialog and returns system replies to individual turns.

        :param dialog:
        :return:
        """
        s_tm1 = self.params['mgr_init_state']
        tr_tm1 = tuple(np.ones((len(self.vocab), )) for _ in range(self.n_db_keys))
        nlu_tm1 = tuple(np.zeros((len(self.vocab), )) for _ in range(self.n_db_keys))

        res = []
        res_aux = []
        db_res_aux = []
        nlu_aux = []
        tr_aux = []
        s_aux = []
        db_count_aux = []

        dialog_turns = zip(dialog[::2], dialog[1::2])
        for O_t, I_t in dialog_turns:
            # 1. Query the database.
            (db_res_t, db_res_t_aux) = self.dbset.forward(tr_tm1)
            db_dist_t = db_res_t[0]; db_t = db_res_t[1:]
            ((db_count_t, ), db_count_t_aux) = Sum.forward((db_dist_t, ))
            db_count_aux.append(db_count_t_aux)

            # 2. Generate system's output.
            ((O_hat_t, ), O_hat_t_aux) = self.nlg.forward((s_tm1, O_t, 0, ) + nlu_tm1 + tr_tm1 + db_t)
            O_hat_t_aux['lens'] = (len(nlu_tm1), len(tr_tm1), len(db_t), )

            # 3. Process what the user said.
            (nlu_t, nlu_t_aux) = self.nlu.forward((I_t, ))
            h_t = nlu_t[0]

            # 4. Update tracker.
            tr_t = []  # For each slot, update the tracker's state.
            tr_t_aux = []
            for tr_tm1_i, nlu_t_i in zip(tr_tm1, nlu_t[1:]):
                ((tr_t_i, ), tr_t_i_aux) = self.tracker.forward((tr_tm1_i, nlu_t_i, h_t, s_tm1, ))
                tr_t.append(tr_t_i)
                tr_t_aux.append(tr_t_i_aux)

            # 5. Update dialog state.
            ((s_t, ), s_t_aux) = self.mgr.forward((s_tm1, h_t, db_count_t, ))

            # Pass current variables to the next step.
            tr_tm1 = tuple(tr_t)
            s_tm1 = s_t
            nlu_tm1 = nlu_t[1:]

            # Save intermediate variables needed for backward pass.
            res.append(O_hat_t)
            res_aux.append(O_hat_t_aux)
            db_res_aux.append(db_res_t_aux)
            nlu_aux.append(nlu_t_aux)
            tr_aux.append(tr_t_aux)
            s_aux.append(s_t_aux)

        res = tuple(res)

        return (res, Vars(
            dialog_turns=dialog_turns,
            res=res_aux,
            s=s_aux,
            tr=tr_aux,
            nlu=nlu_aux,
            db_count=db_count_aux,
            db_res=db_res_aux
        ))


    def backward(self, aux, dO_hat):
        res = []
        ds_tp1 = np.zeros_like(self.params['mgr_init_state'])
        dtr_tp1 = tuple(np.zeros((len(self.vocab), )) for _ in range(self.n_db_keys))
        dnlu_tp1 = tuple(np.zeros((len(self.vocab), )) for _ in range(self.n_db_keys))

        n_steps = 0
        items = reversed(zip(
                aux['dialog_turns'],
                dO_hat,
                aux['res'],
                aux['db_res'],
                aux['db_count'],
                aux['nlu'],
                aux['tr'],
                aux['s']
        ))
        for (sys, usr), dO_hat_t, dO_hat_t_aux, ddb_res_t_aux, ddb_count_t_aux, dnlu_t_aux, dtr_t_aux, ds_t_aux in items:
            n_steps += 1
            dh_t_lst = []
            d_st_lst = []
            (ds_t, dh_t, ddb_count_t, ) = self.mgr.backward(ds_t_aux, (ds_tp1, ))
            d_st_lst.append(ds_t)
            dh_t_lst.append(dh_t)

            dtr_tm1_lst = []
            dtr_tm1 = []
            dnlu_t_lst = []
            for dtr_tp1_i, dtr_t_i_aux, dnlu_tp1_i in zip(dtr_tp1, dtr_t_aux, dnlu_tp1):
                (dtr_t_i, dnlu_t_i, dh_t, ds_t, ) = self.tracker.backward(dtr_t_i_aux, (dtr_tp1_i, ))
                dtr_tm1.append(dtr_t_i)
                dnlu_t_lst.append(dnlu_t_i + dnlu_tp1_i)
                dh_t_lst.append(dh_t)
                d_st_lst.append(ds_t)
            dtr_tm1_lst.append(dtr_tm1)

            (dI_t, ) = self.nlu.backward(dnlu_t_aux, (sum(dh_t_lst), ) + tuple(dnlu_t_lst))

            # NLG.
            dnlg = self.nlg.backward(dO_hat_t_aux, (dO_hat_t,))
            ds_t, dO_t = dnlg[:2]
            dnlu_tp1, dtr_tm1, ddb_t = self._unwrap(dnlg[3:], dO_hat_t_aux['lens'])
            d_st_lst.append(ds_t)
            dtr_tm1_lst.append(dtr_tm1)

            (ddb_dist_t, ) = Sum.backward(ddb_count_t_aux, (ddb_count_t, ))

            ddb_res_t = (ddb_dist_t, ) + ddb_t
            dtr_tm1 = self.dbset.backward(ddb_res_t_aux, ddb_res_t)
            dtr_tm1_lst.append(dtr_tm1)

            ds_tp1 = sum(d_st_lst)
            dtr_tp1 = tuple(np.zeros((len(self.vocab), )) for _ in range(self.n_db_keys))
            for dtr_tp1_i, dtr_tp1_is in zip(dtr_tp1, zip(*dtr_tm1_lst)):
                dtr_tp1_i[:] = sum(dtr_tp1_is)

            res.append(dI_t)
            res.append(dO_t)

        self.grads['mgr_init_state'] += ds_tp1
        assert len(aux['dialog_turns']) == n_steps

        return res[::-1]



    def _unwrap(self, dy, lens):
        ptr = 0
        res = []
        for len_i in lens:
            res.append(dy[ptr:ptr + len_i])
            ptr += len_i

        assert ptr == len(dy)

        return tuple(res)


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

    def decode(self, Y):
        res = []
        for i in range(len(Y)):
            res.append(self.db.vocab.rev(Y[i].argmax()))

        return res

    def prepare_data_signle(self, (q, a)):
        x_q = self.vocab.words_to_ids(q)
        x_a = self.vocab.words_to_ids(a)

        return (x_q, x_a)



# Why does tracker have just 1 set of parameters?