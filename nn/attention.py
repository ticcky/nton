import numpy as np

from base import ParametrizedBlock
from vars import Vars
from softmax import Softmax
from activs import Tanh
from linear import Dot


class Attention(ParametrizedBlock):
    def __init__(self, n_hidden):
        self.n_hid = n_hidden

        Wy = np.random.randn(n_hidden, n_hidden)
        Wh = np.random.randn(n_hidden, n_hidden)
        w = np.random.randn(n_hidden)

        params = Vars(Wh=Wh, Wy=Wy, w=w)
        grads = Vars(Wh=np.zeros_like(Wh), Wy=np.zeros_like(Wy), w=np.zeros_like(w))

        self.parametrize(params, grads)

    def forward(self, (h_out, g_t, emb_in)):
        Wy = self.params['Wy']
        Wh = self.params['Wh']
        w = self.params['w']

        Y = h_out
        n_inputs = len(h_out)

        ((Wy_apply, ), Wy_aux) = Dot.forward((h_out, Wy, ))
        ((Wh_apply, ), Wh_aux) = Dot.forward((g_t, Wh ))

        Wh_dot_g_t_rep = np.repeat(Wh_apply, n_inputs, axis=0)

        Mx = Wy_apply + Wh_dot_g_t_rep

        ((M, ), M_aux) = Tanh.forward((Mx, ))
        ((Mw, ), Mw_aux) = Dot.forward((M, w))

        MwT = Mw.T
        
        ((alpha, ), alpha_aux) = Softmax.forward((MwT, ))

        alphaT = alpha.T

        query = (emb_in * alphaT).sum(axis=0)

        aux = Vars(
            h_out=h_out,
            g_t=g_t,
            emb_in=emb_in,
            Wy_dot_h_out=Wy_apply,
            Wh_dot_g_t=Wh_apply,
            Wh_aux=Wh_aux,
            Wy_aux=Wy_aux,
            M=M,
            M_aux=M_aux,
            Mw_aux=Mw_aux,
            Mx=Mx,
            MwT=MwT,
            alpha=alpha,
            alpha_aux=alpha_aux
        )

        return ((query, ), aux)

    def backward(self, aux, (dquery, )):
        h_out = aux['h_out']
        g_t = aux['g_t']
        emb_in = aux['emb_in']
        alpha = aux['alpha']
        alpha_aux = aux['alpha_aux']
        MwT = aux['MwT']
        Mw_aux = aux['Mw_aux']
        M = aux['M']
        M_aux = aux['M_aux']
        Mx = aux['Mx']
        w = self.params['w']
        Wh = self.params['Wh']
        Wy = self.params['Wy']
        Wh_aux = aux['Wh_aux']
        Wy_aux = aux['Wy_aux']

        dalphaT = np.dot(dquery, emb_in.T)
        demb_in = np.outer(dquery, alpha).T

        dalpha = dalphaT.T

        (dMwT, ) = Softmax.backward(alpha_aux, (dalpha, ))

        dMw = dMwT.T

        #print 'xxx', Mw_aux['A'].shape, Mw_aux['B'].shape

        (dM, dw) = Dot.backward(Mw_aux, (dMw, ))
        (dMx, ) = Tanh.backward(M_aux, (dM, ))

        dWh_dot_g_t_rep = dMx.sum(axis=0, keepdims=True)

        (dh_out, dWy) = Dot.backward(Wy_aux, (dMx, ))
        (dg_t, dWh) = Dot.backward(Wh_aux, (dWh_dot_g_t_rep, ))

        return (dh_out, dg_t.squeeze(), demb_in)
