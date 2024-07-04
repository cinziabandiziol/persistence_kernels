import numpy as np
import time

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

import Dataset as dt

# Function that computes that evaulates the SW function on (D,E) with D,E two PD
def singular_kernel_features(F, G):
    M = 10
    Diag_F = (F + F[:, ::-1]) / 2
    Diag_G = (G + G[:, ::-1]) / 2

    F = np.vstack((F, Diag_G))
    G = np.vstack((G, Diag_F))
    SW = 0
    theta = -np.pi/2
    s = np.pi/M
    # evaluating SW approximated routine
    for j in range(M):
        v1 = np.dot(F, np.array([[np.cos(theta)], [np.sin(theta)]]))
        v2 = np.dot(G, np.array([[np.cos(theta)], [np.sin(theta)]]))
        v1_sorted = np.sort(v1, axis=0, kind='mergesort')
        v2_sorted = np.sort(v2, axis=0, kind='mergesort')
        SW += np.linalg.norm(v1_sorted-v2_sorted, 1)/M
        theta += s

    return SW

# Function that compute the SW Gram matrix
def matrix_kernel_features_fast(data_array):
    n_train = np.shape(data_array)[0]
    result = np.zeros((n_train,n_train))
    for i in range(n_train):
        for j in range(i+1):
            dgm0 = data_array[i]

            dgm1 = data_array[j]

            kSigma = singular_kernel_features(dgm0, dgm1)
            result[i,j] = kSigma
            result[j,i] = kSigma
        print('Iterazione in matrix_kernel_features_fast: ', i, ' of ', n_train)
    return result

np.random.seed(42)

# --------------------------------------------- SETUP ---------------------------------------------------

###################### DATASET SETUP ######################
dataset_name = input('Insert name of the dataset: ')

####################### KERNEL SETUP #######################
kernel_type = 'SWK'

###################### FILE TO WRITE #######################
FileName = './REPORTS/Results_SWK_' + dataset_name + '.txt'

################# CROSS VALIDATION SETUP ####################
program = np.arange(10)
n_fold = 10
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
eta_values_for_CV = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]    
kernel_params_for_CV = [(c, e) for c in C_values for e in eta_values_for_CV]

# ---------------------------------------------------------------------------------------------------------

# Import data
data = dt.Dataset(dataset_name)
PD = data.X
y = data.y

# Compute Gram Matrix
print('Comutation Gram Matrix - Start')
SW_global_matrix = matrix_kernel_features_fast(PD)
print('Comutation Gram Matrix - End')

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

    # Index of train and test sets
    balanced_train_index, balanced_test_index = data.SplitDataset(rand_state)

    # Selection of rows,columns and labels according to the previous indexes 
    SW_train = SW_global_matrix[balanced_train_index,:][:,balanced_train_index]
    SW_test_train = SW_global_matrix[balanced_test_index,:][:,balanced_train_index]
    y_train = y[balanced_train_index]
    y_test = y[balanced_test_index]

    list_mean = np.zeros(len(kernel_params_for_CV))

    kf = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=None)
    #kf = KFold(n_splits=n_fold, shuffle=False, random_state=None)

    iter = 1
    print('Cross-Validation - Start')
    index = range(np.shape(balanced_train_index)[0])

    for train_index, test_index in kf.split(index, y_train):

        # Splitting of the training matrix in training and test parts for CV according to StratifiedKFold subdivisions
        SW_train_fold = SW_train[train_index,:][:,train_index]
        SW_test_train_fold = SW_train[test_index,:][:,train_index]
        y_train_fold = y_train[train_index]
        y_test_fold = y_train[test_index]

        metric_output = []
        progress = 0

        for param in kernel_params_for_CV:
            
            print('[Kernel: ', kernel_type , ', Rand_state: ', rand_state, '] Iteration CV ', progress+1, '/', len(kernel_params_for_CV))
            C = param[0]
            eta = param[1]

            # Compute the Gram matrix
            gram_SW_train_fold = np.exp(-SW_train_fold / (2 * eta**2))
            gram_SW_test_train_fold = np.exp(-SW_test_train_fold / (2 * eta ** 2))

            # Apply the SVC to precomputed matrix
            clf = SVC(kernel='precomputed', C=C, cache_size=2000)
            # Fit the model
            clf.fit(gram_SW_train_fold, y_train_fold)
            # Make predictions
            y_pred_fold = clf.predict(gram_SW_test_train_fold)
            # Evaluate the predictions
            if data.type == 'balanced':
                acc = accuracy_score(y_test_fold, y_pred_fold)
                metric_output.append(acc)
            else:
                balanced_accuracy = balanced_accuracy_score(y_test_fold, y_pred_fold)
                metric_output.append(balanced_accuracy)
            # print(classification_report(y_test_fold, y_pred_fold))

            progress += 1
        list_mean = list_mean + metric_output
        iter += 1

    arr = np.array(list_mean)/n_fold
    # Select best values for parameters
    best_mean = np.max(arr)
    best_mean_index = np.where(arr == best_mean)[0][0]
    best_C = kernel_params_for_CV[best_mean_index][0]
    best_eta = kernel_params_for_CV[best_mean_index][1]

    best_params_CV.append((best_eta, best_C))

    print('Cross-Validation - End')

    print('Model Evaluation - Start')

    # Compute Gram matrix for training and test sets
    gram_SW_train = np.exp(-SW_train / (2 * best_eta**2))
    gram_SW_test_train = np.exp(-SW_test_train / (2 * best_eta ** 2))    

    # Apply the SVC to precomputed matrix 
    classifier = SVC(kernel='precomputed', C = best_C, cache_size=1000)
    # Fit the model with best parameters
    classifier.fit(gram_SW_train, y_train)
    # Make predictions
    y_pred = classifier.predict(gram_SW_test_train)
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
        #print('Balanced_Accuracy: ', acc)
 
    print('Model Evaluation - End')


print('-----------------------------------------------------------------------------------')
print('============================== Classification report ==============================')
print('KERNEL: ', kernel_type)
print('DATASET: ', data.name)
print('DIM. PD: ', data.d)
print('-----------------------------------------------------------------------------------')
print('F1-score (mean out of %s):' % len(program), np.mean(report['f1_score']))
print('F1-score (std out of %s):' % len(program), np.std(report['f1_score']))

if data.type == 'balanced':
    print('Accuracy (mean out of %s):' % len(program), np.mean(report['accuracy']))
    print('Accuracy (std out of %s):' % len(program), np.std(report['accuracy']))
else:
    print('Balanced Accuracy (mean out of %s):' % len(program), np.mean(report['balanced_accuracy']))
    print('Balanced Accuracy (std out of %s):' % len(program), np.std(report['balanced_accuracy']))    


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