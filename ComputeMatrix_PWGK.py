import numpy as np
import pandas as pd
from scipy.spatial import distance_matrix
import time

import Dataset as dt

########################## SETUP ###########################
dataset_name = input('Insert name of the dataset: ')
rho_list = [0.001, 0.01, 0.1, 1, 10, 100, 1000]
rho_list_str = ['0.001', '0.01', '0.1', '1', '10', '100', '1000']
p_list = [1,5,10,50,100]
p_list_str = ['1','5','10','50','100']
C_w_list = [0.001,0.01,0.1,1]
C_w_list_str = ['0.001','0.01','0.1','1']

C_w_p = [(C_w, p) for C_w in C_w_list for p in p_list]
C_w_p_str = [(C_w_str, p_str) for C_w_str in C_w_list_str for p_str in p_list_str]

# Import data
data = dt.Dataset(dataset_name)
PD = data.X
n = PD.shape[0]

pers = lambda x: x[1] - x[0]

# Compute Matrix with different parameters and write it to file
for k in range(len(C_w_p)):

    C_w = C_w_p[k][0]
    p = C_w_p[k][1]

    C_w_str = C_w_p_str[k][0]
    p_str = C_w_p_str[k][1]

    w_arc = lambda x: np.arctan(C_w * (pers(x)) ** p)

    list_w = []

    for i in range(n):
        D = PD[i]
        print('[Compute weights]: Iteration ', i, ' of ', n)
        list_w.append(np.array([[w_arc(x)] for x in D]))

    matrix = np.zeros((n,n))

    for j_rho in range(len(rho_list)):
        rho = rho_list[j_rho]
        rho_str = rho_list_str[j_rho]
        print('Compute matrix ', len(rho_list)*k+j_rho, '/', len(C_w_p)*len(rho_list))
        for i in range(n):
            t1 = time.perf_counter()
            
            D = PD[i]
            
            for j in range(i+1):
                E = PD[j]
                
                M_DD = np.asfarray(distance_matrix(D,D), dtype=float)
                M_EE = np.asfarray(distance_matrix(E,E), dtype=float)
                M_DE = np.asfarray(distance_matrix(D,E), dtype=float)

                K_DD = np.exp(-M_DD ** 2 / (2 * rho ** 2))
                K_EE = np.exp(-M_EE ** 2 / (2 * rho ** 2))
                K_DE = np.exp(-M_DE ** 2 / (2 * rho ** 2))

                w_D = list_w[i]
                w_E = list_w[j]
                H_norm2 = w_D.T@K_DD@w_D + w_E.T@K_EE@w_E - 2 * w_D.T@K_DE@w_E
                matrix[i,j] = H_norm2[0][0]
                matrix[j,i] = H_norm2[0][0]

        with open('./PWGK_MATRICES/' + dataset_name + '/Matrix_PWGK_rho' + rho_str + '_p' + p_str +'_C_w' + C_w_str + '.npy', 'wb') as f:
            np.save(f, matrix)

print('End job')