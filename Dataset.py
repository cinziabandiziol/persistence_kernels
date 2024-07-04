import numpy as np

from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class Dataset:
    def __init__(self,label):
        self.label = label
        self.CreateLoadDataset()

    def CreateLoadDataset(self):

        if self.label == 'PROTEIN_36':
            self.d = 1
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/PROTEIN_36/PD_PROTEIN_36.npy', allow_pickle = True)
            self.y = np.load('./DATASET/PROTEIN_36/y_PROTEIN_36.npy', allow_pickle = True)
            self.name = str('Protein_36')
            self.labels = self.y.tolist()

        elif self.label == 'SHREC14':
            self.d = 1
            self.type = 'balanced'
            self.X = np.load('./DATASET/SHREC14/PD_SHREC14.npy', allow_pickle = True)
            self.y = np.load('./DATASET/SHREC14/y_SHREC14.npy', allow_pickle = True)
            self.name = str('SHREC14')
            self.labels = self.y.tolist()

        elif self.label == 'DIN_SYS':
            self.d = 1
            self.type = 'balanced'
            self.X = np.load('./DATASET/DIN_SYS/PD_DIN_SYS.npy', allow_pickle = True)
            self.y = np.load('./DATASET/DIN_SYS/y_DIN_SYS.npy', allow_pickle = True)
            self.name = str('Dinamical System')
            self.labels = self.y.tolist()

        elif self.label == 'MNIST':
            self.d = [0,1]
            self.type = 'balanced'
            self.X = np.load('./DATASET/MNIST/PD_MNIST.npy', allow_pickle = True)
            self.y = np.load('./DATASET/MNIST/y_MNIST.npy', allow_pickle = True)
            self.name = str('MNIST')
            self.labels = self.y.tolist()

        elif self.label == 'FMNIST':
            self.d = [0,1]
            self.type = 'balanced'
            self.X = np.load('./DATASET/FMNIST/PD_FMNIST.npy', allow_pickle = True)
            y = np.load('./DATASET/FMNIST/y_FMNIST.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('FMNIST')
            self.labels = self.y.tolist()

        elif self.label == 'MUTAG_PATH':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/MUTAG_PATH/PD_MUTAG_PATH.npy', allow_pickle = True)
            y = np.load('./DATASET/MUTAG_PATH/y_MUTAG_PATH.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('MUTAG_PATH')
            self.labels = self.y.tolist()

        elif self.label == 'PTC_PATH':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/PTC_PATH/PD_PTC_PATH.npy', allow_pickle = True)
            y = np.load('./DATASET/PTC_PATH/y_PTC_PATH.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('PTC_PATH')
            self.labels = self.y.tolist()

        elif self.label == 'BZR_PATH':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/BZR_PATH/PD_BZR_PATH.npy', allow_pickle = True)
            y = np.load('./DATASET/BZR_PATH/y_BZR_PATH.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('BZR_PATH')
            self.labels = self.y.tolist()

        elif self.label == 'ENZYMES_PATH':
            self.d = [0,1]
            self.type = 'balanced'
            self.X = np.load('./DATASET/ENZYMES_PATH/PD_ENZYMES_PATH.npy', allow_pickle = True)
            y = np.load('./DATASET/ENZYMES_PATH/y_ENZYMES_PATH.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('ENZYMES_PATH')
            self.labels = self.y.tolist()

        elif self.label == 'DHFR_PATH':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/DHFR_PATH/PD_DHFR_PATH.npy', allow_pickle = True)
            y = np.load('./DATASET/DHFR_PATH/y_DHFR_PATH.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('DHFR_PATH')
            self.labels = self.y.tolist()

        elif self.label == 'PROTEINS_PATH':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/PROTEINS_PATH/PD_PROTEINS_PATH.npy', allow_pickle = True)
            y = np.load('./DATASET/PROTEINS_PATH/y_PROTEINS_PATH.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('PROTEINS_PATH')
            self.labels = self.y.tolist()

        elif self.label == 'MUTAG_JACC':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/MUTAG_JACC/PD_MUTAG_JACC.npy', allow_pickle = True)
            y = np.load('./DATASET/MUTAG_JACC/y_MUTAG_JACC.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('MUTAG_JACC')
            self.labels = self.y.tolist()

        elif self.label == 'PTC_JACC':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/PTC_JACC/PD_PTC_JACC.npy', allow_pickle = True)
            y = np.load('./DATASET/PTC_JACC/y_PTC_JACC.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('PTC_JACC')
            self.labels = self.y.tolist()

        elif self.label == 'BZR_JACC':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/BZR_JACC/PD_BZR_JACC.npy', allow_pickle = True)
            y = np.load('./DATASET/BZR_JACC/y_BZR_JACC.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('BZR_JACC')
            self.labels = self.y.tolist()

        elif self.label == 'ENZYMES_JACC':
            self.d = [0,1]
            self.type = 'balanced'
            self.X = np.load('./DATASET/ENZYMES_JACC/PD_ENZYMES_JACC.npy', allow_pickle = True)
            y = np.load('./DATASET/ENZYMES_JACC/y_ENZYMES_JACC.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('ENZYMES_JACC')
            self.labels = self.y.tolist()

        elif self.label == 'DHFR_JACC':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/DHFR_JACC/PD_DHFR_JACC.npy', allow_pickle = True)
            y = np.load('./DATASET/DHFR_JACC/y_DHFR_JACC.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('DHFR_JACC')
            self.labels = self.y.tolist()

        elif self.label == 'PROTEINS_JACC':
            self.d = [0,1]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/PROTEINS_JACC/PD_PROTEINS_JACC.npy', allow_pickle = True)
            y = np.load('./DATASET/PROTEINS_JACC/y_PROTEINS_JACC.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('PROTEINS_JACC')
            self.labels = self.y.tolist()

        elif self.label == 'ECG200':
            self.d = [0,1,2]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/ECG200/PD_ECG200.npy', allow_pickle = True)
            y = np.load('./DATASET/ECG200/y_ECG200.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('ECG200')
            self.labels = self.y.tolist()

        elif self.label == 'SONY':
            self.d = [0,1,2]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/SONY/PD_SONY.npy', allow_pickle = True)
            y = np.load('./DATASET/SONY/y_SONY.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('SONY')
            self.labels = self.y.tolist()

        elif self.label == 'DISTAL':
            self.d = [0,1,2]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/DISTAL/PD_DISTAL.npy', allow_pickle = True)
            y = np.load('./DATASET/DISTAL/y_DISTAL.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('DISTAL')
            self.labels = self.y.tolist()

        elif self.label == 'STRAWBERRY':
            self.d = [0,1,2]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/STRAWBERRY/PD_STRAWBERRY.npy', allow_pickle = True)
            y = np.load('./DATASET/STRAWBERRY/y_STRAWBERRY.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('STRAWBERRY')
            self.labels = self.y.tolist()

        elif self.label == 'POWER':
            self.d = [0,1,2]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/POWER/PD_POWER.npy', allow_pickle = True)
            y = np.load('./DATASET/POWER/y_POWER.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('POWER')
            self.labels = self.y.tolist()

        elif self.label == 'MOTE':
            self.d = [0,1,2]
            self.type = 'imbalanced'
            self.X = np.load('./DATASET/MOTE/PD_MOTE.npy', allow_pickle = True)
            y = np.load('./DATASET/MOTE/y_MOTE.npy', allow_pickle = True)
            lab = preprocessing.LabelEncoder()
            y_transformed = lab.fit_transform(y)
            self.y = y_transformed
            self.name = str('MOTE')
            self.labels = self.y.tolist()

        else:
            raise TypeError('Dataset name unknown!!')
    
    def SplitDataset(self, rand_state):

        index = range(np.shape(self.y)[0])
        balanced_train_index, balanced_test_index = train_test_split(index, test_size=0.3, random_state=rand_state, stratify=self.labels)

        return balanced_train_index, balanced_test_index