import numpy as np
import time

from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, accuracy_score, balanced_accuracy_score

from sklearn.model_selection import KFold, StratifiedKFold

import Dataset as dt

np.random.seed(42)
# --------------------------------------------- SETUP ---------------------------------------------------

###################### DATASET SETUP ######################
dataset_name = input('Insert name of the dataset: ')

####################### KERNEL SETUP #######################
kernel_type = 'PWGK'

###################### FILE TO WRITE #######################
FileName = './REPORTS/Results_PWGK_' + dataset_name + '.txt'

################# CROSS VALIDATION SETUP ####################
program = np.arange(10)
n_fold = 10
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
tau_values_for_CV = [0.001, 0.01, 0.1, 1, 10, 100]
rho_values_for_CV = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
rho_values_str = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']
p_values_for_CV = [1,5,10,50,100]
p_values_str = ['1','5','10','50','100']
C_w_values_for_CV = [0.001,0.01,0.1,1]
C_w_values_str = ['0.001','0.01','0.1','1']
kernel_params_for_CV = [(r, C_w, p, t, c) for r in rho_values_for_CV for C_w in C_w_values_for_CV for p in p_values_for_CV for t in tau_values_for_CV for c in C_values]
kernel_params_for_CV_small = [(r, C_w, p) for r in rho_values_for_CV for C_w in C_w_values_for_CV for p in p_values_for_CV]
kernel_params_for_CV_small_str = [(r, C_w, p) for r in rho_values_str for C_w in C_w_values_str for p in p_values_str]
kernel_params_str = [(r, C_w, p, t, c) for r in rho_values_str for C_w in C_w_values_str for p in p_values_str for t in tau_values_for_CV for c in C_values]

# ---------------------------------------------------------------------------------------------------------

# Import data
data = dt.Dataset(dataset_name)
PD = data.X
y = data.y
n = PD.shape[0]


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

    print('Inizio Cross-Validation')
    metric_output = []
    progress = 0

    num_comp_matrix = 0

    kf = StratifiedKFold(n_splits=n_fold, shuffle=False, random_state=None)

    print('Cross-Validation - Start')
    index = range(len(balanced_train_index))

    for i in range(len(kernel_params_for_CV_small)):

        param = kernel_params_for_CV_small[i]
        param_str = kernel_params_for_CV_small_str[i]
        num_comp_matrix = num_comp_matrix + 1
        rho = param[0]
        C_w = param[1]
        p = param[2]
        rho_str = param_str[0]
        C_w_str = param_str[1]
        p_str = param_str[2]

        # Load precomputed Matrix
        FileMatrix = './PWGK_MATRICES/' + dataset_name + '/Matrix_PWGK_rho' + rho_str + '_p' +  p_str + '_C_w' + C_w_str + '.npy'
        matrix = np.load(FileMatrix,allow_pickle = True)
            
        for tau in tau_values_for_CV:

            # Compute Gram Matrix
            kk = np.exp(- matrix / (2 * tau**2))

            # Select of rows and columns according to the train indexes 
            kk_train = kk[balanced_train_index,:][:,balanced_train_index]
                
            for C in C_values:
                
                iter = 1
                print('[Kernel: ', kernel_type , ', Rand_state: ', rand_state, '] Iteration CV ', progress+1, '/', len(kernel_params_for_CV))

                metric_output_fold = []
                for train_index, test_index in kf.split(index, y_train):

                    kernel_train_fold = kk_train[train_index,:][:,train_index]
                    kernel_test_train_fold = kk_train[test_index,:][:,train_index]

                    y_train_fold = y_train[train_index]
                    y_test_fold = y_train[test_index]
                    # Apply the SVC to precomputed matrix
                    clf = SVC(kernel='precomputed', C=C, cache_size=2000)
                    # Fit the model
                    clf.fit(kernel_train_fold, y_train_fold)
                    # Make predictions
                    y_pred_fold = clf.predict(kernel_test_train_fold)

                    if data.type == 'balanced':
                        acc = accuracy_score(y_test_fold, y_pred_fold)
                        metric_output_fold.append(acc)
                    else:
                        acc = balanced_accuracy_score(y_test_fold, y_pred_fold)
                        metric_output_fold.append(acc)
                    t6 = time.perf_counter()

                    iter += 1
                mean = np.mean(metric_output_fold)
                metric_output.append(mean)
                progress += 1

    metric_output_array = np.array(metric_output)

    # Select best values for parameters
    best_mean = np.max(metric_output_array)
    best_mean_index = np.where(metric_output_array == best_mean)[0][0]

    best_params_CV.append(kernel_params_for_CV[best_mean_index])
    best_rho = kernel_params_for_CV[best_mean_index][0]
    best_C_w = kernel_params_for_CV[best_mean_index][1]
    best_p = kernel_params_for_CV[best_mean_index][2]
    best_tau = kernel_params_for_CV[best_mean_index][3]
    best_C = kernel_params_for_CV[best_mean_index][4]
    rho_best_str = kernel_params_str[best_mean_index][0]
    C_w_best_str = kernel_params_str[best_mean_index][1]
    p_best_str = kernel_params_str[best_mean_index][2]
    print('Cross-Validation - End')

    print('Model Evaluation - Start')

    # Load precomputed Matrix related to best paramters
    FileMatrixBest = './PWGK_MATRICES/' + dataset_name + '/Matrix_PWGK_rho' + rho_best_str + '_p' +  p_best_str + '_C_w' + C_w_best_str + '.npy'
    matrix_aux = np.load(FileMatrixBest,allow_pickle = True)

    # Compute Gram Matrix related to best paramters
    total_gram_matrix_model_evaluation = np.exp(- matrix_aux / (2 * best_tau**2))

    kernel_train_model_evaluation = total_gram_matrix_model_evaluation[balanced_train_index,:][:,balanced_train_index]
    kernel_test_train_model_evaluation = total_gram_matrix_model_evaluation[balanced_test_index,:][:,balanced_train_index]

    # Apply the SVC
    classifier = SVC(kernel='precomputed', C = best_C, cache_size=2000)
    # Fit the model with best parameters
    classifier.fit(kernel_train_model_evaluation, y_train)
    # Make predictions
    y_pred = classifier.predict(kernel_test_train_model_evaluation)
    print(classification_report(y_test, y_pred))

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