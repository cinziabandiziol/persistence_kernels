clear all; close all;

matrix = load('PD_SHREC14.mat');
PD = matrix.PD;
dims = size(PD);
n_train = dims(2);

FIM_matrix1 = zeros(n_train,n_train);

% Valori per sigma = 0.001, 0.01, 0.1, 1, 10
sigma = 0.001;

ico = 1;
t1 = tic;

tot_iter = n_train*(n_train+1)/2;

for i=1:n_train
    PD_D = PD(i);
    D = PD_D{1,1};
    for j=1:i
        
        iter_mancanti = tot_iter - ico
        PD_E = PD(j);
        E = PD_E{1,1};
        val = compute_dFIM_distance(D, E, sigma);
        FIM_matrix1(i,j) = val;
        FIM_matrix1(j,i) = val;
        ico = ico + 1;
    end
end

runtime = toc(t1);

disp(['running time: ' num2str(runtime)]);

save FIM_SHREC14_sigma0001.mat FIM_matrix1

