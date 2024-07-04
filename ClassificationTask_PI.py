import numpy as np
import time

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score
from sklearn.model_selection import KFold, StratifiedKFold

import Dataset as dt
import PersImage as persimage

np.random.seed(42)

# --------------------------------------------- SETUP ---------------------------------------------------

###################### DATASET SETUP ######################
dataset_name = input('Insert name of the dataset: ')

####################### KERNEL SETUP #######################
kernel_type = 'PI'

###################### FILE TO WRITE #######################
FileName = './REPORTS/Results_PI_' + dataset_name + '.txt'

################# CROSS VALIDATION SETUP ####################
program = np.arange(10)
n_fold = 10
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
sigma_values_for_CV = [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10]
sigma_values_for_CV_str = ['0000001', '000001', '00001', '0001', '001', '01', '1', '10']

kernel_params_for_CV = [(sigma, C) for sigma in sigma_values_for_CV for C in C_values]
kernel_params_for_CV_str = [(sigma_str, C) for sigma_str in sigma_values_for_CV_str for C in C_values]

# ---------------------------------------------------------------------------------------------------------
# Import data
data = dt.Dataset(dataset_name)
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

    # Index of train and test sets
    balanced_train_index, balanced_test_index = data.SplitDataset(rand_state)

    # Selection of labels according to the previous indexes
    y_train = y[balanced_train_index]
    y_test = y[balanced_test_index]

    metric_output = []
    ico_CV = 1

    # kf = KFold(n_splits=n_fold, shuffle=False, random_state=None)
    kf = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=None)

    print('Cross-Validation - Start')
    index = range(len(balanced_train_index))

    for sigma_str in sigma_values_for_CV_str:

        # Load precomputed PI
        PI_array = persimage.PersImage(dataset_name, sigma_str)
        array = PI_array.LoadPI()

        # Select only elements for training set
        array_train = array[balanced_train_index]

        for C in C_values:

            print('[Kernel: ', kernel_type , ', Rand_state: ', rand_state, '] Iteration CV ', ico_CV, '/', len(sigma_values_for_CV)*len(C_values))
            metric_output_fold = []

            ico_fold = 1

            for train_index, test_index in kf.split(index, y_train):

                array_train_fold = array_train[train_index]
                array_test_fold = array_train[test_index]

                y_train_fold = y_train[train_index]
                y_test_fold = y_train[test_index]

                # Apply the SVC to precomputed matrix
                clf = SVC(kernel='linear', C=C, cache_size=2000)
                # Fit the model
                clf.fit(array_train_fold, y_train_fold)
                # Make predictions
                y_pred_fold = clf.predict(array_test_fold)

                if data.type == 'balanced':
                    acc = accuracy_score(y_test_fold, y_pred_fold)
                    metric_output_fold.append(acc)
                else:
                    balanced_accuracy = balanced_accuracy_score(y_test_fold, y_pred_fold)
                    metric_output_fold.append(balanced_accuracy)
                ico_fold = ico_fold + 1
            mean = np.mean(metric_output_fold)
            metric_output.append(mean)
            ico_CV = ico_CV + 1
        
    metric_output_array = np.array(metric_output)
    # Select best values for parameters
    best_mean = np.max(metric_output_array)
    best_mean_index = np.where(metric_output_array == best_mean)[0][0]
    best_sigma = kernel_params_for_CV[best_mean_index][0]
    best_sigma_str = kernel_params_for_CV_str[best_mean_index][0]
    best_C = kernel_params_for_CV[best_mean_index][1]

    best_params_CV.append((best_sigma, best_C))

    print('Cross-Validation - End')

    print('Model Evaluation - Start')

    # Load precomputed PI related to the best sigma
    PI_array_model_evaluation = persimage.PersImage(dataset_name, best_sigma_str)
    array_model_evaluation = PI_array_model_evaluation.LoadPI()
    
    array_train_model_evaluation = array_model_evaluation[balanced_train_index]
    array_test_model_evaluation = array_model_evaluation[balanced_test_index]

    # Apply the SVC
    classifier = SVC(kernel = 'linear', C = best_C, cache_size=2000)
    # Fit the model with best parameters
    classifier.fit(array_train_model_evaluation, y_train)
    # Make predictions
    y_pred = classifier.predict(array_test_model_evaluation)
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