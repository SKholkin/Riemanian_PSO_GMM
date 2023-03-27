from typing import List
from particle import GMMState, Particle
import numpy as np
from copy import deepcopy
from sklearn.mixture import GaussianMixture

class RiemannianGMMPSO:
    def __init__(self, gmm_init: List[GMMState], params, data) -> None:
        self.n_comp = params['n_comp']
        self.n_particles = params['n_particles']
        self.steps = params['steps']
        self.data = data
        self.best_ll = -np.inf

        self.particles = []
        for gmm_state in gmm_init:
            self.particles.append(Particle(gmm_state, params))

        for particle in self.particles:
            ll = particle.get_ll(self.data)
            if ll > self.best_ll:
                self.best_ll = ll
                self.GB = deepcopy(particle.position)
        
        for particle in self.particles:
            particle.set_GB(self.GB)

    def check_global_best(self):
        for particle in self.particles:
            ll = particle.get_ll(self.data)
            if ll > self.best_ll:
                self.best_ll = ll
                self.GB = deepcopy(particle.position)
        
        for particle in self.particles:
            particle.set_GB(self.GB)

    def step(self):
        
        for i_step in range(self.steps):
            print('Before EM: ', [particle.get_ll(self.data) for particle in self.particles])
            for particle in self.particles:
                particle.run_em(self.data)
            self.check_global_best()
            print('GB:', self.best_ll)
            print('After EM: ', [particle.get_ll(self.data) for particle in self.particles])
            for i in range(5):
                for particle in self.particles:
                    particle.step()

                self.check_global_best()
                print('GB:', self.best_ll)

            # check for global best
            
        # run em 
        # run pso for n steps
