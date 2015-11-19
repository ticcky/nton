from vars import Vars

class Block(object):
    pass


class ParametrizedBlock(Block):
    def parametrize(self, params, grads):
        assert params.var_names == grads.var_names
        self._params = params
        self._grads = grads

    def parametrize_from_layers(self, layers, layer_names):
        params = {}
        grads = {}
        for layer_name, layer in zip(layer_names, layers):
            if isinstance(layer, ParametrizedBlock):
                for param_name in layer.params:
                    key = "%s__%s" % (layer_name, param_name, )
                    params[key] = layer.params[param_name]
                    grads[key] = layer.grads[param_name]
            else:
                assert False, "Layer is not a ParametrizedBlock. Perhaps error?"

        self.parametrize(Vars(**params), Vars(**grads))

    @property
    def params(self):
        return self._params

    @property
    def grads(self):
        return self._grads


class Loss(object):
    pass