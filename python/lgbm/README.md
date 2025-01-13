#### py37torch lightGBM_train.py
#### py37torch original_train.py
notes
on bankcrupcy data
AUC: 94.11
Accuracy: 97.37

mlightgbm_train_bank
AUC: 94.19
Accuracy: 97.47

original_train_bank:
AUC: 86.27
ACC: 97.27626459143968

on tweet local data:
lightgbm_train
AUC: 99.96
Accuracy: 99.40

mlightgbm_train
AUC: 99.97
Accuracy: 99.40

original_train:
AUC: 99.70
Accuracy: 99.70


on movielens data
lightgbm_train_movie:
AUC: 66.64
Accuracy: 57.21
lightgbm_train_movie dist_3_l10
AUC: 82.52
Accuracy: 75.09

mlightgbm_train_movie:
AUC: 67.99
chunksize 1000,
Accuracy: 55.88
AUC: 51.15
Accuracy: 50.68
chunksize 10000,                                                                                 AUC: 79.80
AUC: 79.80
Accuracy: 72.51
chunchsize 100000,
AUC: 80.08
Accuracy: 73.08
chunksize 1000000
AUC: 82.27
Accuracy: 74.82

origin_train:
AUC: 82.22
Accuracy: 74.82