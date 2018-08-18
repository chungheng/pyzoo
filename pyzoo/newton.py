"""
zoo-newton: The zeroth-order optimization algorithm with Newton's method.
"""

import numpy as np
from scipy.optimize import OptimizeResult

from .scd import SCD

class ZOONewton(SCD):
    """
    ZOO-Newton: Zeroth Order Stochastic Coordinate Descent with Coordinate-wise Newton’s Method.

    ZOO-Newton

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
    eta : float, optional
        The maximum step size of the gradient.
    step : float, optional
        The step size of update. The default value is 0.001.
    maxiter : int, optional
        The maximum number of iteration. The default value is 1000.
    disp : boolean, optional
        If true, display progress after each iteration
    """

    def __init__(self, func, bounds, x0=None, eta=0.01, step=0.001,
        maxiter=1000, disp=False):

        super(ZOONewton, self).__init__(func, bounds, x0=x0, step=step,
            maxiter=maxiter, disp=disp)


    def display(self, nit):
        pass

    def compute_delta(self, index):
        """
        compute the amount of update along a randomly picked direction.
        """
        pass

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

    def estimate_hessian(self, index):
        """
        estimate the diretional Hessian of a randomly picked direction.
        """
        self.__f = self.func(self.x)

        step_sq = self.step * self.step
        hessian = (self.__f_plus + self.__f_minus - 2.*self.__f) /s tep_sq

        return hessian
