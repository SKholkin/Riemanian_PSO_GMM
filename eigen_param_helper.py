
from particle import GMMState
from python_example import Givens2Matrix_double as Givens2Matrix
from python_example import QRGivens_double as QRGivens
import numpy as np
from types import SimpleNamespace


def eigh_with_fixed_direction_range(spd_matr):
    eigenvalues, v = np.linalg.eigh(spd_matr)

    base_vector = np.ones_like(v[0])
    for i in range(v.shape[0]):
        cos_phi = np.dot(base_vector, v[:, i])
        if cos_phi > 0:
            v[:, i] = -v[:, i]

    return eigenvalues, v

class GMMStateSpectral:
    def __init__(self, means, weights, eigvals, rotations) -> None:
        self.means = means
        self.weights = weights
        self.eigvals = eigvals
        self.rotations = rotations
        self.n_comp, self.d = means.shape[0], means.shape[1]

    def scatter(self, std):
        delta_means = np.random.normal(0, std * np.mean(self.means), size=(self.n_comp, self.d))
        delta_givens =  np.random.uniform(-np.pi * std, std * np.pi, size=(self.n_comp, int(self.d * (self.d - 1) / 2)))

        eig_val_mean = [np.mean(self.eigvals[i]) for i in range(self.n_comp)]

        delta_eigvals = np.random.uniform(0, std * np.mean(eig_val_mean), size=(self.n_comp, self.d))

        self.means += delta_means
        self.rotations += delta_givens
        self.eigvals += delta_eigvals


def to_linear(gmm_state: GMMState):
    n_comp = gmm_state.precisions.shape[0]
    d = gmm_state.precisions.shape[1]

    eigvals = np.zeros([n_comp, d])
    rotations = np.zeros([n_comp, int(d * (d - 1) / 2)])

    for i in range(gmm_state.precisions.shape[0]):
        prec = gmm_state.precisions[i]
        # eigvals, eigvecs = np.linalg.eigh(prec)

        eigenvalues, v = eigh_with_fixed_direction_range(prec)
        # eigenvalues, v = np.linalg.eigh(cov_matrix_list[i])
        givens_rotations = QRGivens(v).squeeze()
        
        eigvals[i] = eigenvalues
        rotations[i] = givens_rotations
    gmm_spec = GMMStateSpectral(gmm_state.means, gmm_state.weights, eigvals, rotations)
    return gmm_spec
    

def to_manifold(gmm_spec_state: GMMStateSpectral) -> GMMState:

    n_comp, d = gmm_spec_state.means.shape[0], gmm_spec_state.means.shape[1]

    prec_matrices = np.zeros([n_comp, d, d])
    for k in range(n_comp):
            eigvals = gmm_spec_state.eigvals[k]
            givens_rotations = gmm_spec_state.rotations[k]
            v = Givens2Matrix(np.expand_dims(givens_rotations, axis=1))
            spd_matr = v @ np.diag(eigvals) @ v.T
            prec_matrices[k] = spd_matr

    gmm_state_dict = {'means_': gmm_spec_state.means,
                        'weights_': gmm_spec_state.weights, 'precisions_': prec_matrices}
    
    return GMMState(SimpleNamespace(**gmm_state_dict))
