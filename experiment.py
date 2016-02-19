"""Run experiments with NTON."""

import argparse
import collections
import json
import logging
import random
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sbt

import data_caminfo
import nton
import nn
import util
import vocab


class C:
  HEADER = '\033[95m'
  BLUE = '\033[94m'
  GREEN = '\033[92m'
  ORANGE = '\033[93m'
  RED = '\033[91m'
  END = '\033[0m'
  BOLD = '\033[1m'
  UNDERLINE = '\033[4m'


def prepare_vocabs(db_index, db_vals):
  nton_vocab = vocab.Vocab()
  nton_vocab.add('<start>')
  with open("caminfo_vocab.json") as f_in:
    for word in json.loads(f_in.read()):
      nton_vocab.add(word)

  entry_vocab = vocab.Vocab(no_oov_eos=True)

  for key, val in db_index:
    for key_part in key:
      nton_vocab.add(key_part)
    entry_vocab.add(val)

  for cont in db_vals:
    for key, val in cont:
      nton_vocab.add(val)

  return entry_vocab, nton_vocab


def prepare_mapping(query_fields, db_fields, nton_vocab):
  map_slu = []
  map_tr = []
  map_db = []

  for field in query_fields:
    map_slu.append(nton_vocab.add('<slu_%s>' % field))
    map_tr.append(nton_vocab.add('<tr_%s>' % field))

  for field in db_fields:
    map_db.append(nton_vocab.add('<db_%s>' % field))

  return map_slu + map_tr + map_db


class Experiment(object):
  def __init__(self, **kwargs):
    logging.info("Starting NTON experiment.")

    cam_info = data_caminfo.DataCamInfo()
    self.data_train = cam_info.gen_data(test_data=False)
    self.data_test = cam_info.gen_data(test_data=True)

    db_index = cam_info.get_db_for(cam_info.query_fields, "id")

    db_vals = []
    for field in cam_info.fields:
      db_vals.append(cam_info.get_db_for(["id"], field))

    logging.info("Using following fields for querying db: %s" %
                 str(cam_info.query_fields))
    logging.info("Using following fields as db output: %s" %
                 str(cam_info.fields))

    entry_vocab, nton_vocab = prepare_vocabs(db_index, db_vals)
    mapping = prepare_mapping(cam_info.query_fields, cam_info.fields, nton_vocab)

    logging.info("Entry vocab has %d keys." % len(entry_vocab))
    logging.info("NTON vocab has %d keys." % len(nton_vocab))
    logging.info("DB mapping: %s" % str(mapping))

    nton_inst = nton.NTON(
        n_cells=50,
        mgr_h_dims=30,
        n_db_keys=len(cam_info.query_fields),
        db_index=db_index,
        db_contents=db_vals,
        db_mapping=mapping,
        vocab=nton_vocab,
        index_vocab=entry_vocab)

    self.nton_inst = nton_inst

    logging.info("Using ADAM update rule.")
    self.update_rule = nn.Adam(nton_inst.params, nton_inst.grads)
    self.stats = collections.defaultdict(lambda: collections.deque(maxlen=100))
    self.example_number = 0

  def train(self):
    avg_loss = collections.deque(maxlen=20)
    losses = []
    train_wers = []
    train_accs = []
    test_wers = []
    test_accs = []
    eval_index = []
    last_print = 0.0
    while True:
      self.example_number +=1
      train_dialog = next(self.data_train)
      training_dialog_labels = self.nton_inst.get_labels(train_dialog)

      self.nton_inst.grads.zero()
      nn.DEBUG.new_dialog(train_dialog, training_dialog_labels)

      (O_hat, O_hat_aux) = self.nton_inst.forward_dialog(train_dialog)

      if time.time() - last_print > 5.0:
        logging.info("Next dialog. First user utterance: %s", train_dialog[0][1][:20])
        self.print_step(train_dialog, training_dialog_labels, O_hat)
        last_print = time.time()

      dO_hat = []
      for O_hat_t, O_t in zip(O_hat, training_dialog_labels):
        ((loss,), loss_aux) = nn.SeqLoss.forward((O_hat_t[:-1], O_t,))
        (dY,) = nn.SeqLoss.backward(loss_aux, 1.0)
        dO_hat.append(dY)

      self.nton_inst.backward(O_hat_aux, dO_hat)
      self.update_rule.update()

  def print_step(self, train_dialog, O, O_hat):
    utterance_accuracy = []
    logging.info(C.RED + '##################### Example %d ####################' + C.END, self.example_number)
    logging.info('Decoded output:')
    for (sys, usr,), O_hat_t, O_t, DEBUG_external_input, DEBUG_entry_dist, DEBUG_db_count in zip(train_dialog, O_hat, O, nn.DEBUG.get_nlg_external_input(), nn.DEBUG.get_db_entry_dist(), nn.DEBUG.get_db_count()):
      O_hat_t = O_hat_t[:-1]
      utterance = np.zeros(len(O_hat_t))
      utterance_scores = np.zeros(len(O_hat_t))
      for i, O_hat_t_i in enumerate(O_hat_t):
        utterance[i] = np.argmax(O_hat_t_i)
        utterance_scores[i] = O_hat_t_i[utterance[i]]

      utterance_accuracy.append(np.sum(utterance == O_t) * 1.0 / len(O_t))

      utterance = map(self.nton_inst.vocab.rev, utterance)
      utterance_true = sys.split()
      utterance_word_lengths = [max(len(w1), len(w2)) for w1, w2 in zip(utterance, utterance_true)]

      logging.info(C.BOLD +    '   System [T]: %s', ' '.join(('%%%ds' % wlen) % w for w, wlen in zip(utterance_true, utterance_word_lengths)[:16]))
      logging.info(            '   System [P]: %s' + C.END, ' '.join(('%%%ds' % wlen) % w for w, wlen in zip(utterance, utterance_word_lengths)[:16]))
      logging.info(C.BLUE +    '   System [S]: %s' + C.END, ' '.join(('%%%d.3f' % wlen) % w for w, wlen in zip(utterance_scores, utterance_word_lengths)[:16]))
      DEBUG_external_input = DEBUG_external_input.split(', ')
      logging.info(           '          %s',  ', '.join(DEBUG_external_input[:3]))
      logging.info(           '          %s',  ', '.join(DEBUG_external_input[3:6]))
      logging.info(           '            Entry dist: %.2f (%d), DB count: %d', np.sum(DEBUG_entry_dist), np.argmax(DEBUG_entry_dist), DEBUG_db_count)
      logging.info(           '          %s',  ', '.join(DEBUG_external_input[6:]))

      logging.info(C.BOLD + '   User:          %s' + C.END, usr)
      logging.info('')

    mean_utterance_accuracy = np.mean(utterance_accuracy)
    self.stats['utterance_accuracy'].append(mean_utterance_accuracy)
    logging.info('Utterance accuracy: %.3f' % mean_utterance_accuracy)
    logging.info('Mean utterance accuracy: %.3f' % np.mean(self.stats['utterance_accuracy']))


  def plot(self, losses, eval_index, (train_wers, train_accs), (test_wers, test_accs),
           plot_filename):
    """Plot training variables."""
    pal = sbt.color_palette()
    fig, ax1 = plt.subplots(linewidth=1)

    ax2 = ax1.twinx()

    plots = []
    plot_labels = []
    plots.append(ax1.plot(np.array(losses), '-', linewidth=1, color=pal[0])[0])
    plot_labels.append('Perplexity')

    plots.append(
        ax2.plot(eval_index, test_wers, 'o-', label='Test WER', markersize=2,
                 linewidth=1, color=pal[1])[0])
    plot_labels.append('Test WER')
    plots.append(
        ax2.plot(eval_index, test_accs, 'o-', label='Test Acc', markersize=2,
                 linewidth=1, color=pal[2])[0])
    plot_labels.append('Test Acc')

    fig.legend(plots, plot_labels, 'upper right')

    plt.savefig(plot_filename)
    plt.close()


def config_numpy():
  np.set_printoptions(edgeitems=3, infstr='inf',
                      linewidth=200, nanstr='nan', precision=4,
                      suppress=False, threshold=1000,
                      formatter={'float': lambda x: "%.1f" % x})





def eval_nton(nton, emb, vocab, data_label, data, n_examples):
  print '### Evaluation(%s): ' % data_label
  wers = []
  acc = []
  for i in xrange(n_examples):
    x_q, x_a = nton.prepare_data_signle(next(data))

    ((x_q_emb,), _) = emb.forward((x_q,))
    x_q_emb = np.vstack((np.ones(x_q_emb.shape[1]), x_q_emb,))

    ((symbol_dec,), _) = emb.forward(([vocab['[EOS]']],))
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
  del logging.root.handlers[:]
  logging.basicConfig(format='[%(asctime)-15s] %(message)s', level=logging.INFO)

  sbt.set()
  matplotlib.use('Agg')
  random.seed(0)
  np.random.seed(0)
  util.pdb_on_error()

  parser = argparse.ArgumentParser()
  parser.add_argument('--n_cells', type=int, default=50)
  parser.add_argument('--eval_step', type=int, default=1000)

  args = parser.parse_args()

  experiment = Experiment(**vars(args))
  experiment.train()
