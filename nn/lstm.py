"""
This is a batched LSTM forward and backward pass

Based on Karpathy's LSTM: https://gist.github.com/karpathy/587454dc0146a6ae21fc
"""
import numpy as np
from base import ParametrizedBlock
from vars import Vars

from utils import timeit

class LSTM(ParametrizedBlock):
    def __init__(self, n_in, n_out):
        self.n_cells = n_out

        WLSTM = np.random.randn(n_in + n_out + 1, 4 * n_out) / np.sqrt(n_in + n_out)
        WLSTM[0,:] = 0 # initialize biases to zero

        WLSTM[0,n_out:2*n_out] = 3

        params = Vars(WLSTM=WLSTM)
        grads = Vars(WLSTM=np.zeros_like(WLSTM))

        self.parametrize(params, grads)

    def get_init(self):
        return (np.zeros((self.n_cells, )), np.zeros((self.n_cells, )))

    def get_init_grad(self):
        return (np.zeros((self.n_cells, )), np.zeros((self.n_cells, )))

    @timeit
    def forward(self, (x, h0, c0 )):
        """
        X should be of shape (t,b,input_size), where t = length of sequence, b = batch size
        """
        WLSTM = self.params['WLSTM']

        n,b,input_size = x.shape
        d = WLSTM.shape[1] / 4 # hidden size
        #if c0 is None: c0 = np.zeros((b,d))
        #if h0 is None: h0 = np.zeros((b,d))

        # Perform the LSTM forward pass with x as the input
        xphpb = WLSTM.shape[0] # x plus h plus bias, lol
        Hin = np.zeros((n, b, xphpb)) # input [1, xt, ht-1] to each tick of the LSTM
        Hout = np.zeros((n, b, d)) # hidden representation of the LSTM (gated cell content)
        IFOG = np.zeros((n, b, d * 4)) # input, forget, output, gate (IFOG)
        IFOGf = np.zeros((n, b, d * 4)) # after nonlinearity
        C = np.zeros((n, b, d)) # cell content
        Ct = np.zeros((n, b, d)) # tanh of cell content
        for t in xrange(n):
          # concat [x,h] as input to the LSTM
          prevh = Hout[t-1] if t > 0 else h0
          Hin[t,:,0] = 1 # bias
          Hin[t,:,1:input_size+1] = x[t]
          Hin[t,:,input_size+1:] = prevh
          # compute all gate activations. dots: (most work is this line)
          IFOG[t] = Hin[t].dot(WLSTM)
          # non-linearities
          IFOGf[t,:,:3*d] = 1.0/(1.0+np.exp(-IFOG[t,:,:3*d])) # sigmoids; these are the gates
          IFOGf[t,:,3*d:] = np.tanh(IFOG[t,:,3*d:]) # tanh
          # compute the cell activation
          prevc = C[t-1] if t > 0 else c0
          C[t] = IFOGf[t,:,:d] * IFOGf[t,:,3*d:] + IFOGf[t,:,d:2*d] * prevc
          Ct[t] = np.tanh(C[t])
          Hout[t] = IFOGf[t,:,2*d:3*d] * Ct[t]

        cache = {}
        cache['WLSTM'] = WLSTM
        cache['Hout'] = Hout
        cache['IFOGf'] = IFOGf
        cache['IFOG'] = IFOG
        cache['C'] = C
        cache['Ct'] = Ct
        cache['Hin'] = Hin
        cache['c0'] = c0
        cache['h0'] = h0

        aux = Vars(**cache)

        return ((Hout, C), aux)  # TODO: Do proper gradient backward for C

    @timeit
    def backward(self, aux, grads):
          dH = grads[0].copy()
          dC = grads[1].copy()

          WLSTM = aux['WLSTM']
          Hout = aux['Hout']
          IFOGf = aux['IFOGf']
          IFOG = aux['IFOG']
          C = aux['C']
          Ct = aux['Ct']
          Hin = aux['Hin']
          c0 = aux['c0']
          h0 = aux['h0']
          n,b,d = Hout.shape
          input_size = WLSTM.shape[0] - d - 1 # -1 due to bias

          assert dC.shape == C.shape, "%s vs %s" % (dC.shape, C.shape, )
          assert dH.shape == dC.shape, "%s vs %s" % (dH.shape, dC.shape, )

          # backprop the LSTM
          dIFOG = np.zeros(IFOG.shape)
          dIFOGf = np.zeros(IFOGf.shape)
          dWLSTM = np.zeros(WLSTM.shape)
          dHin = np.zeros(Hin.shape)
          dC = np.zeros(C.shape) + dC
          dX = np.zeros((n,b,input_size))
          dh0 = np.zeros((b, d))
          dc0 = np.zeros((b, d))
          dHout = dH

          for t in reversed(xrange(n)):

            tanhCt = Ct[t]
            dIFOGf[t,:,2*d:3*d] = tanhCt * dHout[t]
            # backprop tanh non-linearity first then continue backprop
            dC[t] += (1-tanhCt**2) * (IFOGf[t,:,2*d:3*d] * dHout[t])

            if t > 0:
              dIFOGf[t,:,d:2*d] = C[t-1] * dC[t]
              dC[t-1] += IFOGf[t,:,d:2*d] * dC[t]
            else:
              dIFOGf[t,:,d:2*d] = c0 * dC[t]
              dc0 = IFOGf[t,:,d:2*d] * dC[t]
            dIFOGf[t,:,:d] = IFOGf[t,:,3*d:] * dC[t]
            dIFOGf[t,:,3*d:] = IFOGf[t,:,:d] * dC[t]

            # backprop activation functions
            dIFOG[t,:,3*d:] = (1 - IFOGf[t,:,3*d:] ** 2) * dIFOGf[t,:,3*d:]
            y = IFOGf[t,:,:3*d]
            dIFOG[t,:,:3*d] = (y*(1.0-y)) * dIFOGf[t,:,:3*d]

            # backprop matrix multiply
            dWLSTM += np.dot(Hin[t].transpose(), dIFOG[t])
            dHin[t] = dIFOG[t].dot(WLSTM.transpose())

            # backprop the identity transforms into Hin
            dX[t] = dHin[t,:,1:input_size+1]
            if t > 0:
              dHout[t-1,:] += dHin[t,:,input_size+1:]
            else:
              dh0 += dHin[t,:,input_size+1:]

          self.accum_gradients(dWLSTM)

          return (dX, dh0, dc0)

    def accum_gradients(self, dWLSTM):
          self.grads['WLSTM'] += dWLSTM