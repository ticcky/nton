import numpy as np


def check_finite_differences(fwd_fn, bwd_fn, delta=1e-5, n_times=10, gen_input_fn=None, test_inputs=(0, )):
    """Check that the analytical gradient `bwd_fn` matches the true gradient.
    It is verified using the finite differences method on fwd_fn..
    :param fwd_fn:
    :param bwd_fn:
    :param delta:
    :param n_times:
    :param extra_args:
    :param gen_input_fn:
    :returns True if all gradient checks were ok, False if some failed
    """
    assert gen_input_fn != None

    for n in range(n_times):
        rand_input = gen_input_fn()

        (y, ), out_aux = fwd_fn(rand_input)
        dy = (np.random.randn(*y.shape), )

        grads = bwd_fn(rand_input, out_aux, dy)

        for i in test_inputs:
            assert grads[i].shape == rand_input[i].shape, "shape1=%s, shape2=%s" % (grads[i].shape,  rand_input[i].shape, )

            for dim, x in enumerate(rand_input[i].flat):
                orig = rand_input[i].flat[dim]
                rand_input[i].flat[dim] = orig + delta
                out1 = (fwd_fn(rand_input)[0] * dy[0]).sum()
                rand_input[i].flat[dim] = orig - delta
                out2 = (fwd_fn(rand_input)[0] * dy[0]).sum()
                rand_input[i].flat[dim] = orig

                grad_num = ((out1 - out2) / (2 * delta))
                grad_an = grads[i].flat[dim]

                if abs(grad_num) < 1e-7 and abs(grad_an) < 1e-7:
                    pass
                else:
                    rel_error = abs(grad_an - grad_num) / abs(grad_an + grad_num)
                    if rel_error > 1e-2:
                        print 'GRADIENT WARNING', 'inp', i, 'dim', dim, 'val', x
                        print 'analytic', grad_an, 'num', grad_num
                        print 'rel error', rel_error
                        if rel_error > 1:
                            print 'GRADIENT ERROR TOO LARGE!'
                            return False

    return True


class TestParamGradInLayer:
    """Wrap given layer into a layer that accepts its parameter as input.
    Useful for gradient checking by check_finite_differences."""
    def __init__(self, layer, param_name, layer_input):
        self.param_name = param_name
        self.layer = layer
        self.orig_shape = layer.params[param_name].shape
        self.layer_input = layer_input

    def gen(self):
        return np.random.randn(*self.orig_shape)

    def forward(self, (x, )):
        self.layer.params[self.param_name][:] = x
        return self.layer.forward(self.layer_input)

    def backward(self, (x, ), aux, grads):
        assert np.allclose(self.layer.params[self.param_name][:], x)

        self.layer.grads.zero()

        self.layer.backward(self.layer_input, aux, grads)

        return (self.layer.grads[self.param_name], )