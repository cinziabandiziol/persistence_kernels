clear all; close all;

matrix = load('PD_PROTEIN_36.mat');
PD = matrix.PD;
n_train = 1357;

FIM_matrix1 = zeros(n_train,n_train);

% Valori per sigma = 0.001, 0.01, 0.1, 1, 10
sigma = 0.001;

ico = 1;
t1 = tic;

tot_iter = n_train*(n_train+1)/2;

for i=1:n_train
    D = [cell2mat(PD(i,:,1))' cell2mat(PD(i,:,2))'];

    for j=1:i
        
        iter_mancanti = tot_iter - ico
        E = [cell2mat(PD(j,:,1))' cell2mat(PD(j,:,2))'];
        val = compute_dFIM_distance(D, E, sigma);
        FIM_matrix1(i,j) = val;
        FIM_matrix1(j,i) = val;
        ico = ico + 1;
    end
end

runtime = toc(t1);

disp(['running time: ' num2str(runtime)]);

save FIM_PROTEIN_36_sigma0001.mat FIM_matrix1

