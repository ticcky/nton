from nn import ParametrizedBlock, Vars, Sequential, LinearLayer, Softmax, Switch, Concat
from db_dist import DBDist


class Tracker(ParametrizedBlock):
    def __init__(self, input_h_size, input_s_size):
        self.mlp_update = Sequential([
            LinearLayer(input_h_size + input_s_size, 2),
            Softmax(),
        ])

        self.parametrize_from_layers([self.mlp_update], ["mlp_update"])

    def forward(self, (tr, slu, h_t, s, )):
        ((mlp_in, ), mlp_in_aux, ) = Concat.forward((h_t, s))
        ((p_update, ), p_update_aux, ) = self.mlp_update.forward((mlp_in, ))

        ((tr_prime, ), tr_prime_aux) = Switch.forward((p_update, tr, slu, ))

        return ((tr_prime, ), Vars(
            mlp_in=mlp_in_aux,
            p_update=p_update_aux,
            tr_prime=tr_prime_aux
        ))

    def backward(self, aux, (dtr_prime, )):
        (dp_update, dtr, dslu, ) = Switch.backward(aux['tr_prime'], (dtr_prime, ))

        (dmlp_in, ) = self.mlp_update.backward(aux['p_update'], (dp_update, ))
        (dh_t, ds, ) = Concat.backward(aux['mlp_in'], (dmlp_in, ))

        return (dtr, dslu, dh_t, ds, )







