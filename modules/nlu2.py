import numpy as np

from nn import LSTM, ParametrizedBlock, Attention, Vars


class NLU2(ParametrizedBlock):
  def __init__(self, n_cells, emb_dim, n_slu):
    self.input_rnns = []
    self.slus = []
    for i in range(n_slu):
      self.input_rnns.append(LSTM(n_in=emb_dim, n_out=n_cells))
      self.slus.append(Attention(n_hidden=n_cells))

    self.parametrize_from_layers(
        self.input_rnns + self.slus,
        ["input_rnn%.2d" % i for i in range(n_slu)] +
        ["slu%.2d" % i for i in range(n_slu)]
    )

  def forward(self, (E, )):
    input_rnn_aux = []
    slu_res = []
    slu_aux = []
    lst_h_t = []
    for input_rnn, slu in zip(self.input_rnns, self.slus):
      h0, c0 = input_rnn.get_init()
      ((H, C), H_aux) = input_rnn.forward((E[:, np.newaxis, :], h0, c0,))
      input_rnn_aux.append(H_aux)
      H = H[:, 0]
      C = C[:, 0]

      h_t = H[-1]
      lst_h_t.append(h_t)

      ((slu_i,), slu_i_aux) = slu.forward((H, h_t, E))
      slu_res.append(slu_i)
      slu_aux.append(slu_i_aux)

    h_t = sum(lst_h_t)
    return ((h_t,) + tuple(slu_res), Vars(
        slu=slu_aux,
        input_rnn=input_rnn_aux
    ))

  def backward(self, aux, grads):
    assert type(grads) == tuple
    assert len(grads) == len(self.slus) + 1

    dh_t_1 = grads[0]
    dslus = grads[1:]

    lst_dE = []
    for input_rnn, input_rnn_aux, slu, dslu_i, slu_i_aux in zip(self.input_rnns, aux['input_rnn'], self.slus, dslus, aux['slu']):
      dH, dh_t, dE_1 = slu.backward(slu_i_aux, (dslu_i,))

      dH[-1] += dh_t + dh_t_1
      dH = dH[:, np.newaxis, :]
      dC = np.zeros_like(dH)

      (dE_2, _, _) = input_rnn.backward(input_rnn_aux, (dH, dC))

      lst_dE.append(dE_2[:, 0, :] + dE_1)


    dE = sum(lst_dE)

    return (dE,)
