from vars import Vars

class Block(object):
    def accum_grads(self, accums, grads):
        assert len(grads) == len(accums)

        for grad, accum in zip(grads, accums):
            accum.append(grad)


class ParametrizedBlock(Block):
    def parametrize(self, params, grads):
        assert params.var_names == grads.var_names
        self._params = params
        self._grads = grads

    def parametrize_from_layers(self, layers, layer_names, extra_params=None, extra_grads=None):
        assert len(layers) == len(layer_names)
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

        if extra_params:
          assert extra_grads
          assert not set(params.keys()).intersection(extra_params.keys())
          assert set(extra_params.keys()) == set(extra_grads.keys())

          params.update(extra_params)
          grads.update(extra_grads)

        self.parametrize(Vars(**params), Vars(**grads))

    @property
    def params(self):
        return self._params

    @property
    def grads(self):
        return self._grads


class Loss(object):
    pass