# Code repository of the preprint:
# Persistence kernels for classification : A comparative study

## **Abstract**

The aim of the present work is a comparative study of different persistence kernels applied to various classification problems. After some necessary preliminaries on homology and persistence diagrams, we introduce five different kernels that are then used to compare their performances of classification on various datasets. We also provide the Python codes for the reproducibility of results.

## **Software requirements**

Code runs with python 3.11.x. The libraries needed are:

- numpy 1.21.4
- scipy 1.7.2
- scikit-learn 1.0.1
- numba 0.58.1
- persim 0.3.5

and for computed PFK matrices Matlab R2023b.

## **How to run**

In DATASET, the user can find the precomputed Persistence Diagrams with a file with corresponding labels. With the aim to reduce costs in terms of time, we preferred precomputed also a preGram matrix for some kernels and the PI. We describe here how to proceed.

1. Compute the preGram matrices for kernels
     - PSSK: the user first create a folder with the name of dataset in PSSK_MATRICES, and then runs _python ComputeMatrix_PSSK.py_ and then write to the shell the name of the dataset of interest
     - PWGK: the user first create a folder with the name of dataset in PWGK_MATRICES, and then runs _python ComputeMatrix_PWGK.py_ and then write to the shell the name of the dataset of interest
     - PI: the user first create a folder with the name of dataset in PERSISTENCE_IMAGES, and then runs _python ComputePI.py_ and then write to the shell the name of the dataset of interest
  
2. Chose 1 kernel, for example kernel = SWK, and run
   
   $ python ClassificationTask_SWK.py
   
   and then insert the name of the dataset of interest. After the job ends, a file will be created in folder REPORT summarizing the information about dataset and the results of classification.
