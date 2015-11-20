import numpy as np

from vars import Vars


class VanillaSGD(object):
    def __init__(self, params, grads, lr=0.1):
        self.params = params
        self.grads = grads
        self.lr = lr

    def update(self):
        self.params.increment_by(self.grads, factor=-self.lr)


class AdaMax(object):
    #def __init__(self, params, grads, alpha=0.002, beta1=0.9, beta2=0.999):
    def __init__(self, params, grads, alpha=0.0002, beta1=0.1, beta2=0.001):
        self.params = params
        self.grads = grads

        self.m = Vars.create_from(params)
        self.u_comp = Vars.create_from(params)
        self.u = 0.0
        self.t = 0

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2

    def update_var(self, theta, g, m):
        m[:] = self.beta1 * m + (1 - self.beta1) * g
        theta[:] = theta[:] - (self.alpha / (1 - np.power(self.beta1, self.t))) * m / self.u

    def update(self):
        self.t += 1
        g_norm = 0.0
        for param_name in self.params:
            g_norm += np.sum(self.grads[param_name]**2)
        g_norm = np.sqrt(g_norm)

        self.u = max(self.beta2 * self.u, g_norm)
        for param_name in self.params:
            self.update_var(
                theta=self.params[param_name],
                g=self.grads[param_name],
                m=self.m[param_name]
            )


class Adam(object):
    def __init__(self, params, grads, alpha=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        #def __init__(self, params, grads, alpha=0.0002, beta1=0.1, beta2=0.001, eps=1e-8):
        self.params = params
        self.grads = grads

        self.m = Vars.create_from(params)
        self.v = Vars.create_from(params)
        self.t = 0

        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def update_var(self, theta, g, m, v):
        m[:] = self.beta1 * m + (1 - self.beta1) * g
        v[:] = self.beta2 * v + (1 - self.beta2) * g**2

        mhat = m / (1 - np.power(self.beta1, self.t))
        vhat = v / (1 - np.power(self.beta2, self.t))

        theta[:] = theta[:] - (self.alpha * mhat / (np.sqrt(vhat) + self.eps))

    def update(self):
        self.t += 1

        for param_name in self.params:
            self.update_var(
                theta=self.params[param_name],
                g=self.grads[param_name],
                m=self.m[param_name],
                v=self.v[param_name]
            )