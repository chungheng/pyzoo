"""
zoo-adam: The zeroth-order optimization algorithm with ADAM.
"""

import numpy as np
from scipy.optimize import OptimizeResult

from .scd import SCD

class ADAM(object):
    """
    An optimizor for the gradient descent algorithm on multivariate space.

    ADAM stands for ``adaptive moment estimation''.

    Parameters
    ----------
    dim : int
        The dimensionality of the varaible space.
    eta : float, optional
        The maximum step size of the gradient.
    beta1 : float, optional
        The scaling term of the first momentum.
    beta2 : float, optional
        The scaling term of the second momentum.
    epsilon : float, optional
        The bias term for avoiding zero value at the denominator.
    """
    def __init__(self, dim, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.T = np.zeros(dim, dtype=np.int)
        self.M = np.zeros(dim)
        self.v = np.zeros(dim)
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

    def update(self, index, gradient):
        self.T[index] += 1
        self.M[index] = self.beta1*self.M[index] + (1.-self.beta1)*gradient
        self.v[index] = self.beta2*self.v[index] + (1.-self.beta2)*gradient*gradient

        M_hat = self.M[index] / (1.-pow(self.beta1, self.T[index]))
        v_hat = self.v[index] / (1.-pow(self.beta2, self.T[index]))

        delta = -self.eta*M_hat / (np.sqrt(v_hat)+self.epsilon)
        return delta

class ZOOADAM(SCD):
    """
    Zeroth Order Stochastic Coordinate Descent with Coordinate-wise ADAM.

    ZOO-ADAM

    Parameters
    ----------
    func : callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.
    bounds : sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.
    x0 : tuple, optional
        Initial value of variables. If not given, ``x0`` will be randomly
        initialized.
    step : float, optional
        The step size for estimating diretional gradient. The default value is
        0.001.
    maxiter : int, optional
        The maximum number of iteration. The default value is 1000.
    disp : boolean, optional
        If true, display progress after each iteration
    eta : float, optional
        The maximum step size of the gradient.
    beta1 : float, optional
        The scaling term of the first momentum.
    beta2 : float, optional
        The scaling term of the second momentum.
    epsilon : float, optional
        The bias term for avoiding zero value at the denominator.
    """

    def __init__(self, func, bounds, x0=None, step=0.001, maxiter=1000,
        disp=False, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):

        super(ZOOADAM, self).__init__(func, bounds, x0=x0, step=step,
            maxiter=maxiter, disp=disp)

        adam_params = adam_params or dict()
        self.adam = ADAM(self.dim, **adam_params)

    def display(self, nit):
        print "[Error] Plus: %.5f Minus: %.5f" % (self.__f_plus, self.__f_minus)

    def compute_delta(self, index):
        """
        compute the amount of update along a randomly picked direction.
        """
        gradient = self.estimate_gradient(index)
        delta = self.adam.update(index, gradient)

        return delta

    @property
    def result(self):
        """
        return the optimization result.
        """
        return OptimizeResult(x=self.x, nit=self.maxiter, nfev=2*self.maxiter,
                              fun=0.5*(self.__f_plus+self.__f_minus))

    def estimate_gradient(self, index):
        """
        estimate the partial derivative of a randomly picked direction.
        """
        perturbation = self.basis[index]*self.step
        x_plus = self._scale_parameters(self.__unscaled_x + perturbation)
        x_minus = self._scale_parameters(self.__unscaled_x - perturbation)

        self.__f_plus = self.func(x_plus)
        self.__f_minus = self.func(x_minus)

        gradient = (self.__f_plus - self.__f_minus) / (2.*self.step)

        return gradient
