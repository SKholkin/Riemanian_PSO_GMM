from sklearn.preprocessing import MinMaxScaler
import os.path as osp
import pandas as pd
import numpy as np
from math import copysign, hypot
from sklearn.datasets import load_digits
from sklearn.mixture import GaussianMixture

real_dataset_names = ['breast_cancer', 'cloud']

def load_breast_cancer():
    bc_data_name = 'data/wdbc.data'
    with open(bc_data_name) as f:
        data_bc = pd.DataFrame([item.split(',')[2:] for item in f.readlines()])
    data_bc = data_bc.astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(data_bc)
    data_bc = scaler.transform(data_bc)
    return data_bc

def load_cloud_dataset():
    cloud_data_name = 'data/cloud.data'
    with open(cloud_data_name) as f:
        cloud_data = pd.DataFrame([item.split() for item in f.readlines()])
    cloud_data = cloud_data.astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(cloud_data)
    cloud_data = scaler.transform(cloud_data)
    return cloud_data

def load_satelite_dataset():
    f_name = osp.join('data', 'sat.trn')
    with open(f_name) as f:
        sat_data = pd.DataFrame([item.split()[:-1] for item in f.readlines()])
    sat_data = sat_data.astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(sat_data)
    sat_data = scaler.transform(sat_data)
    return sat_data[:int(sat_data.shape[0] / 4), :]

def load_seg_data():
    f_name = osp.join('data', 'segmentation.data')
    with open(f_name) as f:
        sat_data = pd.DataFrame([item.split(',')[1:] for item in f.readlines()])
    sat_data = sat_data[6:].astype(float).to_numpy()
    scaler = MinMaxScaler()
    scaler.fit(sat_data)
    sat_data = scaler.transform(sat_data)
    return sat_data

def load_synthetic_dataset(filename):
    data = np.load(filename)
    scaler = MinMaxScaler()
    scaler.fit(data)
    data = scaler.transform(data)
    return data

def load_usps_dataset():
    import h5py
    filename = 'data/usps.h5'
    with h5py.File(filename, 'r') as hf:
            train = hf.get('train')
            X_tr = train.get('data')[:]
            y_tr = train.get('target')[:]
            test = hf.get('test')
            X_te = test.get('data')[:]
            y_te = test.get('target')[:]
    print(np.array(X_tr).shape)

def load_digits_dataset():
    return load_digits().data

def load_dataset(dataset_name):
    if dataset_name == 'breast_cancer':
        return load_breast_cancer()
    if dataset_name == 'cloud':
        return load_cloud_dataset()
    if dataset_name == 'landsat':
        return load_satelite_dataset()
    if dataset_name == 'seg':
        return load_seg_data()
    if dataset_name == 'digits':
        return load_digits_dataset()
    if osp.isfile(dataset_name):
        return load_synthetic_dataset(dataset_name)
    raise RuntimeError(f'Unknown dataset {dataset_name}. Please provide path to synthetic dataset file or correctly write real dataset name')


from particle import GMMState
from eigen_param_helper import to_linear, to_manifold
from copy import deepcopy
def init_gmm_via_delta(params, data):
    n_iter = params['em_init_iters']
    gmm = GaussianMixture(n_components=params['n_comp'], covariance_type='full', n_init=params['n_particles'], max_iter=n_iter, init_params='k-means++')
    gmm.fit(data)
    print('Basic GMM score:', gmm.score(data))
    gmm_state = GMMState(gmm)
    ret_val = []
    for i in range(params['n_particles']):
        gmm_state_copy = deepcopy(gmm_state)
        gmm_spectral_state = to_linear(gmm_state_copy)
        # print(gmm_spectral_state.eigvals)
        gmm_spectral_state.scatter(0.03)

        ret_val.append(to_manifold(gmm_spectral_state))

    return ret_val

if __name__ == '__main__':
    params = {'n_comp': 10, 'n_particles': 50, 'steps': 5, 
              'mu': 0, 'r_1': 0.5, 'r_2': 0.8,
               'r_1_w': 0.42, 'r_2_w': 0.57, 'em_init_iters': 100}
    
    data = load_dataset('cloud')
    from particle import Particle
    list_of_gmmstates = init_gmm_via_delta(params, data)
    print([Particle(item, params).get_ll(data) for item in list_of_gmmstates])
    # print(list_of_gmmstates[0].precisions)
    