import numpy as np
from persim import PersistenceImager

import Dataset as dt

########################## SETUP ###########################
dataset_name = input('Insert name of the dataset: ')
sigma_list = [0.000001, 0.00001, 0.00001, 0.001, 0.01, 0.1, 1, 10]
sigma_list_str = ['0000001', '000001', '00001', '0001', '001', '01', '1', '10']

root = './PERSISTENCE_IMAGES/' + dataset_name + '/'

# Import data
data = dt.Dataset(dataset_name)
PD = data.X

for k in range(len(sigma_list)):

    sigma = sigma_list[k]
    sigma_str = sigma_list_str[k]
    print('Computation of PI for sigma ', sigma_str,' - Start')

    # Computation of PI using persim library
    pimgr = PersistenceImager(pixel_size=0.1, weight='linear_ramp', weight_params = {},
                            kernel_params={'sigma': [[sigma, 0.0], [0.0, sigma]]})

    pimgr.fit(PD, skew=True)
    pimgs_new = pimgr.transform(PD, skew=True)

    list_array = []

    # Flat the PD
    for i in range(len(pimgs_new)):
        arr_local = pimgs_new[i]
        arr_flat = arr_local.flatten()
        list_array.append(arr_flat)

    PI_array = np.array(list_array)

    print('Computation of PI for sigma ', sigma_str,' - End')

    with open(root + 'PI_' + dataset_name + '_sigma' + sigma_str + '.npy', 'wb') as f:
        np.save(f, PI_array)

print('End job')