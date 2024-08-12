# external imports
import numpy as np
import time
import pprint
from scipy.linalg import solve

# internal imports
from pipt.loop.assimilation import Assimilate
from pipt.misc_tools import analysis_tools as at


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
        method: str
            Method for optimization. Default is 'Levenberg-Marquardt'
        bounds: list
            Bounds for the optimization
        options: dict
            Available options:
            - maxiter: int
                Maximum number of iterations
        """


        super().__init__(**options)
        def __set__variable(var_name=None, default=None):
            if var_name in options:
                return options[var_name]
            else:
                return default

        self.fun = fun
        self.x = x
        self.args = args
        self.kg = kalmangain
        self.bounds = bounds

        # Extract values or set to default
        self.maxiter = __set__variable('maxiter', 100)
        self.data_misfit_tol = __set__variable('data_misfit_tol', 0.01)
        self.step_tol = __set__variable('step_tol', 0.01)
        self.lam = __set__variable('lam', 100)
        self.lam_max = __set__variable('lam_max', 1e10)
        self.lam_min = __set__variable('lam_min', 0.01)
        self.gamma = __set__variable('gamma', 5)
        self.energy = __set__variable('energy', 0.95)
        # check that the energy is below 1. If not, divide by 100.
        if self.energy > 1:
            self.energy /= 100.
        self.iteration = __set__variable('iteration', 1)

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

        if self.method == 'LM':
            #todo:
            # Evaluate if we should add an option for different update schemes under the LM umbrella.
            # Example of such schemes are the approximate LM, the full LM, the subspace LM, or perhaps the
            # Square-root LM.

            self.step = at.calc_LM_step(self.aug_pred_data, self.scale_data, self.cov_data, self.lam)
            self.state = self.current_state + self.step
        elif self.method == 'GN':
            self.step = at.calc_GN_step(self.aug_pred_data, self.scale_data, self.cov_data, self.lam)
            self.state = self.current_state + self.gamma*self.step
        else:
            raise ValueError(f'Update method not {self.method} recognized.')

        if 'localanalysis' in self.keys_da:
            self.local_analysis_update()
        else:
            # Mean pred_data and perturbation matrix with scaling
            if len(self.scale_data.shape) == 1:
                self.pert_preddata = np.dot(np.expand_dims(self.scale_data ** (-1), axis=1),
                                            np.ones((1, self.ne))) * np.dot(self.aug_pred_data, self.proj)
            else:
                self.pert_preddata = solve(
                    self.scale_data, np.dot(self.aug_pred_data, self.proj))

            aug_state = at.aug_state(self.current_state, self.list_states)
            self.update()  # run ordinary analysis
            if hasattr(self, 'step'):
                aug_state_upd = aug_state + self.step
            if hasattr(self, 'w_step'):
                self.W = self.current_W + self.w_step
                aug_prior_state = at.aug_state(self.prior_state, self.list_states)
                aug_state_upd = np.dot(aug_prior_state, (np.eye(
                    self.ne) + self.W / np.sqrt(self.ne - 1)))

            # Extract updated state variables from aug_update
            self.state = at.update_state(aug_state_upd, self.state, self.list_states)
            self.state = at.limits(self.state, self.prior_info)