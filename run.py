
from pso import RiemannianGMMPSO

if __name__ == '__main__':
    params = {'n_comp': 10, 'n_particles': 50, 'steps': 5, 
              'mu': 0, 'r_1': 0.5, 'r_2': 0.8,
               'r_1_w': 0.42, 'r_2_w': 0.57, 'em_init_iters': 100}
    
    pso = RiemannianGMMPSO(params)
