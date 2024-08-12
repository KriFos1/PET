# external imports
import numpy as np
import time
import pprint
from scipy.linalg import solve

# internal imports
from pipt.loop.assimilation import Assimilate
from pipt.misc_tools import analysis_tools as at
from utils.common import vectorize

class EnRML(Assimilate):
    def __init__(self, fun, x, args, kalmangain, bounds=None, **options):
        """
        Parameters
        ----------
        fun: callable
            Objective function
        x: ndarray
            Initial guess
        args: tuple
            Arguments to pass to the objective function
        kalmangain: callable
            Gain matrix to be applied.
        bounds: dict
            Bounds for the optimization, given as a dictionary with keys as states and values as tuples with lower
            and higher bounds.
        options: dict
            Available options:
            - maxiter: int
                Maximum number of iterations
        """


        super().__init__(**options)

        self.fun = fun
        self.x = x
        self.args = args
        self.kg = kalmangain
        self.bounds = bounds

        # Extract values or set to default
        self.maxiter = options.get('maxiter', 100)
        self.data_misfit_tol = options.get('data_misfit_tol', 0.01)
        self.step_tol = options.get('step_tol', 0.01)
        self.lam = options.get('lam', 100)
        self.lam_max = options.get('lam_max', 1e10)
        self.lam_min = options.get('lam_min', 0.01)
        self.gamma = options.get('gamma', 5)
        self.energy = options.get('energy', 0.95)
        # check that the energy is below 1. If not, divide by 100.
        if self.energy > 1:
            self.energy /= 100.
        self.iteration = options.get('iteration', 1)

        if self.logger is not None:
            self.logger.info('       ====== Running Data Assimilation - EnRML ======')
            self.logger.info('\n' + pprint.pformat(self.options))

        # The EnOpt class self-ignites, and it is possible to send the EnOpt class as a callale method to scipy.minimize
        self.run_loop()  # run_loop resides in the Optimization class (super)

    def calc_update(self):
        """
        Calculate the update step in LM-EnRML, which is just the Levenberg-Marquardt update algorithm with
        the sensitivity matrix approximated by the ensemble.
        """

        if self.iteration == 1:  # first iteration
            data_misfit = at.calc_objectivefun(
                self.real_obs_data, self.aug_pred_data, self.cov_data)

            # Store the (mean) data misfit (also for conv. check)
            self.data_misfit = np.mean(data_misfit)
            self.prior_data_misfit = np.mean(data_misfit)
            self.data_misfit_std = np.std(data_misfit)

            if self.lam == 'auto':
                self.lam = (0.5 * self.prior_data_misfit) / self.aug_pred_data.shape[0]

            self.logger.info(
                f'Prior run complete with data misfit: {self.prior_data_misfit:0.1f}. '
                f'Lambda for initial analysis: {self.lam}')
        else:
            # Mean pred_data and perturbation matrix with scaling
            if len(self.scale_data.shape) == 1:
                self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                            np.ones((1, self.ne))) * np.dot(self.aug_pred_data, self.proj)
            else:
                self.pert_preddata = solve(
                    self.scale_data, np.dot(self.aug_pred_data, self.proj))

            step = self.kg(self.aug_pred_data, self.scale_data, self.proj, self.lam)

            # Extract updated state variables from aug_update
            self.state = self.state + step
            self.state = at.limits(self.state, self.bounds)