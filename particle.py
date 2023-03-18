from types import SimpleNamespace
import numpy as np
from copy import deepcopy
from spd_ai_geometry import SPDManifold
from sklearn.mixture import GaussianMixture


class GMMState:
    def __init__(self, gmm) -> None:
        self.means = gmm.means_
        self.weights = gmm.weights_
        self.precisions = gmm.precisions_
        self.n_comp = self.means.shape[0]

    def zeros_like(self):
        gmm_state_dict = {'means_': np.zeros_like(self.means),
                          'weights_': np.zeros_like(self.weights), 'precisions_': self.precisions_}
        return GMMState(SimpleNamespace(**gmm_state_dict))
    
    def log_spd(self, Sigma):
        copy = deepcopy(self)
        for i in range(self.n_comp):
            copy.precisions[i] = SPDManifold.Log(Sigma, copy.precisions[i])
        return copy
    
    def exp_spd(self, Sigma):
        copy = deepcopy(self)
        for i in range(self.n_comp):
            copy.precisions[i] = SPDManifold.Exp(Sigma, copy.precisions[i])
        return copy
    
    def __add__(self, gmm_2):
        self.means += gmm_2.means
        self.weights += gmm_2.weights
        self.precisions += gmm_2.precisions
        return self


class Particle:
    def __init__(self, init_gmm, pso_params) -> None:
        self.mu = pso_params['mu']
        self.r_1, self.r_2 = pso_params['r_1'], pso_params['r_2']
        self.r_1_w, self.r_2_w = pso_params['r_1_w'], pso_params['r_2_w']
        self.position = GMMState(init_gmm)
        self.velocity = self.position.zeros_like()
        self.PB = GMMState(init_gmm)
        self.GB = GMMState(init_gmm)

    def set_GB(self, GB: GMMState):
        self.GB = GB

    def run_em(self, data):
        raise NotImplementedError

    def step(self):
        velocity_projected = self.velocity.log_spd()
        pb_projected = self.PB.log_spd()
        gb_projected = self.GB.log_spd()
        self._step_euclidian(velocity_projected, pb_projected, gb_projected)
        self.position = self.position + self.velocity
        self.position = self.position.exp_spd()

    def get_ll(self, data):

        gmm = GaussianMixture(n_components=self.position.weights.shape[0], covariance_type='full', weights_init=self.position.weights, means_init=self.position.means, precisions_init=self.position.precisions, max_iter=0)

        cholesky = np.zeros_like(self.position.precisions)
        
        for i in range(self.n_components):
            cholesky[i] = np.linalg.cholesky(self.position.precisions[i])

        gmm.weights_ = self.position.weights
        gmm.means_ = self.position.means
        gmm.precisions_cholesky_ = cholesky

        return gmm.score(data)

    def _step_euclidian(self, velocity, PB, GB):
        c_1 = np.random.uniform(0, 1)
        c_2 = np.random.uniform(0, 1)
        self.velocity.weights = self.mu * velocity.weights + c_1 * self.r_1_w * (PB.weights - self.position.weights) +c_2 * self.r_2_w * (GB.weights - self.position.weights)
        self.velocity.means = self.mu * velocity.means + c_1 * self.r_1 * (PB.means - self.position.means) +c_2 * self.r_2 * (GB.means - self.position.means)
        self.velocity.precisions = self.mu * velocity.precisions + c_1 * self.r_1 * (PB.precisions - self.position.precisions) +c_2 * self.r_2 * (GB.precisions - self.position.precisions)   
