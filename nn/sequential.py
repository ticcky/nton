import numpy as np
import time
import random

#from nn import Embeddings, Linear, Sigmoid, Softmax, CrossentropyLoss, RNN, Params, ParametrizedBlock, Rectify
#from dataset import Dataset
#from nn.tanh import Tanh
from base import ParametrizedBlock
from vars import Vars


class Sequential(ParametrizedBlock):
    """Chains several layer one on top of each other."""
    def __init__(self, layers):
        self.layers = layers

        params = {}
        grads = {}
        for i, layer in enumerate(layers):
            if isinstance(layer, ParametrizedBlock):
                for param_name in layer.params:
                    key = "%2d__%s" % (i, param_name, )
                    params[key] = layer.params[param_name]
                    grads[key] = layer.grads[param_name]

        self.parametrize(Vars(**params), Vars(**grads))

    def forward(self, (x, )):
        ycache = []
        yaux = []
        last_y = x
        for layer in self.layers:
            ((y, ), y_aux) = layer.forward((last_y, ))

            ycache.append(y)
            yaux.append(y_aux)

            last_y = y

        aux = Vars(
            ycache=ycache,
            yaux=yaux
        )

        return ((last_y, ), aux)

    def backward(self, (x, ), aux, (dy, )):
        ycache = aux['ycache']
        yaux = aux['yaux']

        last_dy = dy
        for i, layer in reversed(list(enumerate(self.layers))):
            if i > 0:
                last_in = ycache[i - 1]
            else:
                last_in = x
            last_out = ycache[i]
            last_aux = yaux[i]
            #print type(last_in), type(last_dy)
            (last_dy, ) = layer.backward((last_in, ),  last_aux, (last_dy, ))

        return (last_dy, )