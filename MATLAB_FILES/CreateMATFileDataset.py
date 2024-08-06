import numpy as np

from scipy.io import savemat

dataset_name = input('Insert the name of the dataset: ')

# CREATE MAT FILE FROM npy DATASET
PD = np.load('./../DATASET/' + dataset_name + '/PD_' + dataset_name + '.npy', allow_pickle = True)

dictionary = {"PD": PD}
savemat("./PD_PTC.mat", dictionary)

print('Save ' + dataset_name + ' dataset')