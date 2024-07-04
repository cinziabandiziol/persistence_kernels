import numpy as np
import pandas as pd

class PersImage:
    def __init__(self, dataset_name, sigma_str):
        self.dataset_name = dataset_name
        self.sigma_str = sigma_str

    def LoadPI(self):
        data = np.load('./PERSISTENCE_IMAGES/' + self.dataset_name + '/PI_' + self.dataset_name + '_sigma' + self.sigma_str + '.npy')
        return data