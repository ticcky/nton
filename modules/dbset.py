from nn import Block, Vars
from db_dist import DBDist


class DBSet(Block):
    def __init__(self, index_content, contents, vocab, entry_vocab):
        self.index_db = DBDist(index_content, vocab, entry_vocab)

        self.content_dbs = []
        for content in contents:
            self.content_dbs.append(DBDist(content, entry_vocab, vocab))

    def forward(self, inputs):
        assert type(inputs) == tuple
        assert len(inputs) == self.index_db.n

        ((entry_dist, ), entry_dist_aux) = self.index_db.forward(inputs)

        res = []; res_aux = []
        for content_db in self.content_dbs:
            ((db_val, ), db_val_aux) = content_db.forward((entry_dist, ))
            res.append(db_val); res_aux.append(db_val_aux)

        return (tuple(res), Vars(
            res=res_aux,
            entry_dist=entry_dist_aux
        ))

    def backward(self, aux, dres):
        lst_dentry_dist = []
        for ddb_val, res_aux, content_db in zip(dres, aux['res'], self.content_dbs):
            self.accum_grads((lst_dentry_dist, ), content_db.backward(res_aux, (ddb_val, )))

        dentry_dist = sum(lst_dentry_dist)

        dinputs = self.index_db.backward(aux['entry_dist'], (dentry_dist, ))

        return dinputs







