from typing import List
from particle import GMMState, Particle
import numpy as np
from copy import deepcopy
from sklearn.mixture import GaussianMixture

def init_gmm_via_delta(params, data):
    n_iter = params['em_init_iters']
    gmm = GaussianMixture(n_components=params['n_comp'], covariance_type='full', max_iter=n_iter)
    gmm.fit(data)

    self.means = gmm.means_
    self.weights = gmm.weights_
    self.precisions = gmm.precisions_

    


class RiemannianGMMPSO:
    def __init__(self, gmm_init: List[GMMState], params, data) -> None:
        self.n_comp = params['n_comp']
        self.n_particles = params['n_particles']
        self.steps = params['steps']
        self.data = data

        self.particles = []
        for gmm_state in gmm_init:
            self.particles.append(Particle(gmm_state, params))

        for particle in self.particles:
            ll = particle.get_ll(self.data)
            if ll > self.best_ll:
                self.best_ll = ll
                self.GB = deepcopy(particle)

    def check_global_best(self):
        for particle in self.particles:
            ll = particle.get_ll(self.data)
            if ll > self.best_ll:
                self.best_ll = ll
                self.GB = deepcopy(particle)

    def step(self):
        for i_step in range(self.steps):
            for particle in self.particles:
                particle.run_em(self.data)
            for particle in self.particles:
                particle.step()

            self.check_global_best()

            # check for global best
            
        # run em 
        # run pso for n steps
