
from pso import RiemannianGMMPSO
from utils import load_dataset, init_gmm_via_delta
import numpy as np
import pandas as pd

if __name__ == '__main__':
    params = {'n_comp': 10, 'n_particles': 50, 'steps': 5, 
              'mu': 0, 'r_1': 0.5, 'r_2': 0.8,
               'r_1_w': 0.42, 'r_2_w': 0.57, 'em_init_iters': 100}
    
    # make an intial GMM 
    # transform it to GMMstate
    # scatter in spectral parametrization
    # pass into PSO
    n_runs = 1
    res_list = pd.DataFrame(columns=['ll'])
    for i in range(n_runs):
        data = load_dataset('seg')
        gmm_states = init_gmm_via_delta(params, data)
        pso = RiemannianGMMPSO(gmm_states, params, data)
        print('PSO best intial LL:', pso.best_ll)
        pso.step()
        best_ll = pso.best_ll

        res_list = pd.concat([res_list, pd.DataFrame({'ll': best_ll}, index=[1])],  ignore_index=True)

    from datetime import datetime

    res_list.to_csv(f'res_{datetime.now().isoformat(timespec="seconds")}')
    print('Result: ', np.mean(res_list['ll']), '+-' ,  np.std(res_list['ll']))
