from nn import Block, Vars, Sequential, LinearLayer, Tanh, Concat


class Manager(Block):
    def __init__(self, input_h_size, input_s_size, db_count_size, hidden_size):
        self.mlp_update = Sequential([
            LinearLayer(input_h_size + input_s_size + db_count_size, hidden_size),
            Tanh(),
            LinearLayer(hidden_size, input_s_size)
        ])

    def forward(self, (s, h_t, db_count, )):
        ((mlp_in, ), mlp_in_aux, ) = Concat.forward((s, h_t, db_count, ))
        ((s_prime, ), s_prime_aux, ) = self.mlp_update.forward((mlp_in, ))

        return ((s_prime, ), Vars(
            mlp_in=mlp_in_aux,
            s_prime=s_prime_aux
        ))

    def backward(self, aux, (ds_prime, )):
        (dmlp_in, ) = self.mlp_update.backward(aux['s_prime'], (ds_prime, ))
        (ds, dh_t, ddb_count, ) = Concat.backward(aux['mlp_in'], (dmlp_in, ))

        return (ds, dh_t, ddb_count, )
