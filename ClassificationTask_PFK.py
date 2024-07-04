import numpy as np
import time

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from scipy.spatial import distance_matrix
from sklearn.model_selection import KFold, StratifiedKFold

import Dataset as dt
import MatrixFIM as mfim

np.random.seed(42)

# --------------------------------------------- SETUP ---------------------------------------------------

###################### DATASET SETUP ######################
dataset_name = input('Insert name of the dataset: ')

####################### KERNEL SETUP #######################
kernel_type = 'PFK'

###################### FILE TO WRITE #######################
FileName = './REPORTS/Results_PFK_' + dataset_name + '.txt'

################# CROSS VALIDATION SETUP ####################
program = np.arange(10)
n_fold = 10
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
sigma_values_for_CV = [0.001, 0.01, 0.1, 1, 10]
t_values_for_CV = [0.1, 1, 10, 100, 1000]
kernel_params_for_CV = [(sigma, C, t) for sigma in sigma_values_for_CV for C in C_values for t in t_values_for_CV]
Ct_params_for_CV = [(C, t) for C in C_values for t in t_values_for_CV]

# ---------------------------------------------------------------------------------------------------------
# Import data
data = dt.Dataset(dataset_name, [1])
y = data.y

if data.type == 'balanced':
    report = {'f1_score': [], 'accuracy': []}
else:
    report = {'f1_score': [], 'balanced_accuracy': []}

best_params_CV = []

for rand_state in program:

    print('======================')
    print(kernel_type)
    print('======================')
    print('Iteration number: ', rand_state + 1, ' of ', len(program))

    print('Cross-Validation - Start')

    # Index of train and test sets
    balanced_train_index, balanced_test_index = data.SplitDataset(rand_state)

    # Selection of labels according to the previous indexes
    y_train = y[balanced_train_index]
    y_test = y[balanced_test_index]

    kf = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=None)
    # kf = KFold(n_splits=n_fold, shuffle=False, random_state=None)

    metric_output = []
    ico_CV = 1

    print('Cross-Validation - Start')
    index = range(len(balanced_train_index))

    for sigma in sigma_values_for_CV:

        # Load precomputed FIM Matrix
        matrixFIM = mfim.MatrixFIM(dataset_name, sigma)
        matrix = matrixFIM.LoadMatrix()

        # Select only elements for training set
        kk_train = matrix[balanced_train_index,:][:,balanced_train_index]

        for params_CV in Ct_params_for_CV:

            C = params_CV[0]
            t = params_CV[1]

            metric_output_fold = []

            print('[Kernel: ', kernel_type , ', Rand_state: ', rand_state, '] Iteration CV ', ico_CV, '/', len(kernel_params_for_CV))

            for train_index, test_index in kf.split(index, y_train):

                matrix_train_fold = kk_train[train_index,:][:,train_index]
                matrix_test_train_fold = kk_train[test_index,:][:,train_index]

                kernel_train_fold = np.exp(-matrix_train_fold / t)
                kernel_train_test_fold = np.exp(-matrix_test_train_fold / t)

                y_train_fold = y_train[train_index]
                y_test_fold = y_train[test_index]

                # Apply the SVC to precomputed matrix
                clf = SVC(kernel='precomputed', C=C, cache_size=2000)
                # Fit the model
                clf.fit(kernel_train_fold, y_train_fold)
                # Make predictions 
                y_pred_fold = clf.predict(kernel_train_test_fold)
                
                if data.type == 'balanced':
                    acc = accuracy_score(y_test_fold, y_pred_fold)
                    metric_output_fold.append(acc)
                else:
                    balanced_accuracy = balanced_accuracy_score(y_test_fold, y_pred_fold)
                    metric_output_fold.append(balanced_accuracy)

            mean = np.mean(metric_output_fold)
            metric_output.append(mean)
            ico_CV = ico_CV + 1
        
    metric_output_array = np.array(metric_output)

    # Select best values for parameters
    best_mean = np.max(metric_output_array)
    best_mean_index = np.where(metric_output_array == best_mean)[0][0]
    best_sigma = kernel_params_for_CV[best_mean_index][0]
    best_C = kernel_params_for_CV[best_mean_index][1]
    best_t = kernel_params_for_CV[best_mean_index][2]

    best_params_CV.append((best_sigma, best_t, best_C))

    print('Cross-Validation - End')

    print('Model Evaluation - Start')

    # Load precomputed FIM Matrix related to the best sigma
    matrixFIM_model_evaluation = mfim.MatrixFIM(dataset_name, best_sigma)
    matrix_model_evaluation = matrixFIM_model_evaluation.LoadMatrix()
    
    # Select only some rows and columns from the FIM Matrix
    kernel_train_model_evaluation = matrix_model_evaluation[balanced_train_index,:][:,balanced_train_index]
    kernel_test_train_model_evaluation = matrix_model_evaluation[balanced_test_index,:][:,balanced_train_index]

    gram_PFK_total_train = np.exp(- kernel_train_model_evaluation / best_t)
    gram_PFK_total_test_train = np.exp(- kernel_test_train_model_evaluation / best_t)

    # Apply the SVC
    classifier = SVC(kernel = 'precomputed', C = best_C, cache_size=2000)
    # Fit the model with best parameters
    classifier.fit(gram_PFK_total_train, y_train)
    # Make predictions
    y_pred = classifier.predict(gram_PFK_total_test_train)
    print(classification_report(y_test, y_pred))

    # compute and save measures to evaluate performances of the model: accuracy and F1 score
    f1_s = f1_score(y_test, y_pred, average='macro')
    report['f1_score'] = report['f1_score'] + [f1_s]

    if data.type == 'balanced':
        acc = accuracy_score(y_test, y_pred)
        report['accuracy'] = report['accuracy'] + [acc]
        #print('Accuracy: ', acc)
    else:
        acc = balanced_accuracy_score(y_test, y_pred)
        report['balanced_accuracy'] = report['balanced_accuracy'] + [acc]
        #print('balanced_accuracy: ', acc)  

    print('Model Evaluation - End')

f = open(FileName,'w')

# Write final results on screen and on file in folder REPORTS
f.write('-----------------------------------------------------------------------------------')
f.write('\n')
f.write('============================== Classification report ==============================')
f.write('\n')
f.write('KERNEL: ')
f.write(kernel_type)
f.write('\n')
f.write('DATASET: ')
f.write(data.name)
f.write('\n')
f.write('DIM. PD: ')
f.write(str(data.d))
f.write('\n')

f.write('-----------------------------------------------------------------------------------')
f.write('\n')
f.write('F1-score (mean out of 10): ')
f.write(str(np.mean(report['f1_score'])))
f.write('\n')
f.write('F1-score (std out of 10): ')
f.write(str(np.std(report['f1_score'])))
f.write('\n')

if data.type == 'balanced':
    f.write('Accuracy (mean out of 10): ')
    f.write(str(np.mean(report['accuracy'])))
    f.write('\n')
    f.write('Accuracy (std out of 10): ')
    f.write(str(np.std(report['accuracy'])))
else:
    f.write('Balanced Accuracy (mean out of 10): ')
    f.write(str(np.mean(report['balanced_accuracy'])))
    f.write('\n')
    f.write('Balaced Accuracy (std out of 10): ')
    f.write(str(np.std(report['balanced_accuracy'])))

f.close()

print('End job')