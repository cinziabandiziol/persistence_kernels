from scipy.io import loadmat

class MatrixFIM:
    def __init__(self, dataset_name, sigma):
        self.dataset_name = dataset_name
        self.sigma = sigma
        str_sigma = str(sigma)
        self.str_sigma = str_sigma.replace('.','')

    def LoadMatrix(self):

        File = './PFK_MATRICES/' + self.dataset_name + '/FIM_' + self.dataset_name + '_sigma' + self.str_sigma + '.mat'
        data = loadmat(File)
        matrix = data['FIM_matrix1']
        
        return matrix