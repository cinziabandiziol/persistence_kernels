import numpy as np
import math
from numba import jit

import Dataset as dt

# Function that computes that evaulates the SW function on (D,E) with D,E two PD
#@jit(nopython=True)
def singular_kernel_features(pd1, pd2, sigma):
    kSigma = 0.0

    for k in range(pd1.shape[0]):
        y = pd1[k]
        for l in range(pd2.shape[0]):
            z = pd2[l]
            yz = (y[0] - z[0])**2 + (y[1] - z[1])**2
            yz_bar = (y[0] - z[1])**2 + (y[1] - z[0])**2
            kSigma += math.exp(-(yz) / (8 * sigma)) - math.exp(-(yz_bar) / (8 * sigma))

    kSigma = kSigma/(8 * np.pi * sigma)
    return kSigma

# Function that compute the SW Gram matrix
def matrix_kernel_features_fast(data_array, sigma):
    n_train = np.shape(data_array)[0]
    result = np.zeros((n_train,n_train))
    for i in range(n_train):
        for j in range(i+1):
            dgm0 = data_array[i]
            dgm1 = data_array[j]
            kSigma = singular_kernel_features(dgm0, dgm1, sigma)
            result[i,j] = kSigma
            result[j,i] = kSigma
    return result

########################## SETUP ###########################
dataset_name = input('Insert name of the dataset: ')
sigma_list = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
sigma_list_str = ['0000001', '000001', '00001', '0001', '001', '01', '1', '10']

# Import data
data = dt.Dataset(dataset_name, [1])
PD = data.X

for k in range(len(sigma_list)):

    sigma = sigma_list[k]
    sigma_str = sigma_list_str[k]
    print('Compute matrix ', k+1, ' of ', len(sigma_list))
    total_gram_matrix = matrix_kernel_features_fast(PD, sigma)

    Filename = './PSSK_MATRICES/' + dataset_name + '/Matrix_PSSK_sigma' + sigma_str + '_' + dataset_name + '.npy'

    with open(Filename, 'wb') as f:
        np.save(f, total_gram_matrix)

print('End job')