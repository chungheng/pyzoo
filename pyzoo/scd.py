"""
scd: The Stochastic Coordinate Descent Algorithm.
"""

import numpy as np

class SCD(object):
    """
    Stochastic Coordinate Descent.
    """

    def __init__(self, func, bounds, x0=None, step=0.001, maxiter=1000,
        disp=False, **kwargs):
        self.maxiter = maxiter
        self.dim = len(bounds)
        self.func = func
        self.bounds = np.array(bounds, dtype='float').T
        self.basis = np.eye(self.dim)
        self.step = step

        if (np.size(self.bounds, 0) != 2 or not
                np.all(np.isfinite(self.bounds))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        self.__scale_arg1 = 0.5 * (self.bounds[0] + self.bounds[1])
        self.__scale_arg2 = np.fabs(self.bounds[0] - self.bounds[1])

        if x0 is not None:
            self.__unscaled_x = self._unscale_parameters(x0)
        else:
            self.__unscaled_x = np.random.rand(self.dim)

        self.disp = disp

    def display(self, nit):
        """
        print the progress of otimization

        Parameters:
        nit : int
            Index of the current iteration.
        """
        pass

    @property
    def x(self):
        """
        return scaled x.
        """
        return self._scale_parameters(self.__unscaled_x)

    @property
    def result(self):
        """
        return the optimization result.
        """
        pass

    @abstractmethod
    def compute_delta(self, index):
        """
        compute the amount of update along a randomly picked direction.
        """
        pass

    def solve(self):
        """
        aprroximate the local minimum over iterations.
        """
        for i in xrange(self.maxiter):
            index = np.random.randint(self.dim)
            delta = self.compute_delta(index)
            self.__unscaled_x[index] += delta

            if self.disp:
                self.display()

        return self.result

    def _scale_parameters(self, trial):
        """
        scale from a number between 0 and 1 to parameters.
        """
        return self.__scale_arg1 + (trial - 0.5) * self.__scale_arg2

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5
