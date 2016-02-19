import numpy as np

from nn import ParametrizedBlock, Vars, Sequential, LinearLayer, Tanh, Concat


class Manager(ParametrizedBlock):
    def __init__(self, input_h_size, state_size, db_count_size, hidden_size):
        self.state_size = state_size

        self.mlp_update = Sequential([
            LinearLayer(input_h_size + state_size + db_count_size, hidden_size),
            Tanh(),
            LinearLayer(hidden_size, state_size)
        ])

        self.parametrize_from_layers(
            [self.mlp_update], ["mlp_update"]
        )

    def get_zero_state(self):
        return np.zeros((self.state_size,))

    def forward(self, (s, h_t, db_count_0, db_count_1, db_count_2p )):
        ((mlp_in, ), mlp_in_aux, ) = Concat.forward((s, h_t, db_count_0, db_count_1, db_count_2p, ))
        ((s_prime, ), s_prime_aux, ) = self.mlp_update.forward((mlp_in, ))

        return ((s_prime, ), Vars(
            mlp_in=mlp_in_aux,
            s_prime=s_prime_aux
        ))

    def backward(self, aux, (ds_prime, )):
        (dmlp_in, ) = self.mlp_update.backward(aux['s_prime'], (ds_prime, ))
        (ds, dh_t, ddb_count_0, ddb_count_1, ddb_count_2p, ) = Concat.backward(aux['mlp_in'], (dmlp_in, ))

        return (ds, dh_t, ddb_count_0, ddb_count_1, ddb_count_2p)
