import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os
import random
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from matplotlib import pyplot as plt
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, classification_report, confusion_matrix, accuracy_score, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import MinMaxScaler

tf.set_random_seed(3333)
run_flg = 1
sess_save_nam = './save/'
confusion_file_nam = './results/confusion.txt'
train_file_nam = './Selected_Data/train_file.txt'
test_file_nam = './Selected_Data/test_file.txt'
train_final_file_nam = './results/train_result.txt'
test_final_file_nam = './results/test_result.txt'

# 1. Hyperparameter
n_trn_rate = 0.8
cl = 100  # Uranium = 30
dim_output = 2
n_iteration = 50
LN_dist = [8, 10]

# Normalization (0 ~ 1) of data
def normalization(dat):
    minv = np.min(dat)
    maxv = np.max(dat)
    dat = (dat - minv) / (maxv - minv)
    return dat, minv, maxv

if run_flg == 0:
    # 2. Data
    dat_file_nam = './Data/Radon.txt'  # file_nam = Radon & Uranium
    dat_raw = np.loadtxt(dat_file_nam).astype(np.float32)

    ##### Original dataset and Random-undersampling #####
    n_excss_tgt = np.sum(dat_raw[:, 3:4] >= cl)
    is_excss = np.reshape([dat_raw[:, 3:4] >= cl], len(dat_raw), 1)
    is_non_excss = np.reshape([dat_raw[:, 3:4] < cl], len(dat_raw), 1)
    excss_dat = dat_raw[is_excss, :]
    non_excss_dat = dat_raw[is_non_excss, :]
    for ii in range(n_iteration):
        shff_no = np.random.permutation(non_excss_dat.shape[0])
        non_excss_dat = non_excss_dat[shff_no, :]
        non_excss_dat = non_excss_dat[:len(excss_dat), :]
        dat = np.concatenate((excss_dat, non_excss_dat), axis=0)

        tgt = dat[:, 3:4]  # target variable (Rn)
        g = dat[:, 1:2]  # geology type
        expl = dat[:, 4:]  # applied explanatory variables
        loc_idx = dat[:, 0:1]  # location index
        n_dat = np.shape(dat)[0]
        n_trn = np.int(np.floor(n_trn_rate * n_dat))
        n_tst = n_dat - n_trn
        n_rock = np.int(np.max(dat[:, 1]))  # number of rock type


        expl_norm = np.zeros(np.shape(expl))
        for i in range(np.shape(expl)[1]):
            expl_norm[:, i], min_expl, max_expl = normalization(expl[:, i])

        # dummy variable
        dum = np.zeros([len(g), np.int(np.max(g)) - 1])
        for i in range(len(g)):
            if g[i] == 1:  # Granite
                dum[i, :] = [0, 0, 0, 0]
            elif g[i] == 2:  # Volcanic
                dum[i, :] = [1, 0, 0, 0]
            elif g[i] == 3:  # Metamorphic
                dum[i, :] = [0, 1, 0, 0]
            elif g[i] == 4:  # Granitic Metamorphic
                dum[i, :] = [0, 0, 1, 0]
            elif g[i] == 5:  # Sedimentary
                dum[i, :] = [0, 0, 0, 1]


        # Target variable (Rn) 1: Rn>=100, 0: Rn<100
        T = np.zeros([len(tgt), dim_output])
        for i in range(len(tgt)):
            if tgt[i] >= cl:
                T[i, :] = [1, 0]
            else:
                T[i, :] = [0, 1]

        # other X variables (T, pH, Eh, EC, DO) + dummy variables
        XX = np.concatenate((dum, expl_norm), axis=1)
        shff_no = np.random.permutation(XX.shape[0])
        IN = XX[shff_no, :]
        OUT = T[shff_no, :]
        loc_idx = loc_idx[shff_no]
        tgt = tgt[shff_no]
        x_train, x_test = IN[:n_trn, :], IN[n_trn:, :]
        y_train, y_test = OUT[:n_trn, :], OUT[n_trn:, :]

        dim_input = np.shape(IN)[1]

        rnd_clf = RandomForestClassifier(n_estimators=300, max_leaf_nodes=100, n_jobs=-1)
        rnd_clf.fit(x_train, y_train[:, 0:1])
        y_pred = rnd_clf.predict_proba(x_test)
        prd_label = np.zeros([len(y_pred), 1])
        for m in range(len(y_pred)):
            if y_pred[m, 0] >= y_pred[m, 1]:
                prd_label[m] = 0
            elif y_pred[m, 0] < y_pred[m, 1]:
                prd_label[m] = 1

        con = confusion_matrix(y_test[:, 0], prd_label)
        print("=" * 100)
        print(con)
        print("Accuracy is {} %.".format(sum(np.diag(con)) / sum(sum(con)) * 100))
        if sum(np.diag(con)) / sum(sum(con)) * 100 > 72:
            train_mat = np.concatenate((loc_idx[:n_trn], tgt[:n_trn], x_train), axis=1)
            test_mat = np.concatenate((loc_idx[n_trn:], tgt[n_trn:], x_test, y_pred, prd_label), axis=1)
            np.savetxt(train_file_nam, train_mat)
            np.savetxt(test_file_nam, test_mat)

elif run_flg == 1:
    # 2. Data
    train_raw = np.loadtxt(train_file_nam).astype(np.float32)  # loc, tgt[0/1], explanatory
    test_raw = np.loadtxt(test_file_nam).astype(np.float32)  #loc, tgt[0/1], explanatory, prob., prd_label
    dat_raw = np.concatenate((train_raw, test_raw[:, :-3]), axis=0)

    n_trn = np.shape(train_raw)[0]
    n_tst = np.shape(test_raw)[0]
    n_dat = np.shape(dat_raw)[0]
    n_rock = 5  # number of rock type

    tgt = dat_raw[:, 1:2]
    T = np.zeros([len(tgt), dim_output])
    for i in range(len(tgt)):
        if tgt[i] >= cl:
            T[i, :] = [1, 0]
        else:
            T[i, :] = [0, 1]

    IN = dat_raw[:, 2:]
    OUT = T
    loc_idx = dat_raw[:, 0:1]
    x_train, x_test = IN[:n_trn, :], IN[n_trn:, :]
    y_train, y_test = OUT[:n_trn, :], OUT[n_trn:, :]

    dim_input = np.shape(IN)[1]
    con_sum = np.zeros([2, 2])
    for mm in range(n_iteration):

        rnd_clf = RandomForestClassifier(n_estimators=300, max_leaf_nodes=70, n_jobs=-1)
        rnd_clf.fit(x_train, y_train[:, 0:1])
        y_pred = rnd_clf.predict_proba(x_test)
        prd_label = np.zeros([len(y_pred), 1])
        for m in range(len(y_pred)):
            if y_pred[m, 0] >= y_pred[m, 1]:
                prd_label[m] = 0
            elif y_pred[m, 0] < y_pred[m, 1]:
                prd_label[m] = 1

        con = confusion_matrix(y_test[:, 0], prd_label)
        con_sum = con_sum + con
        print("=" * 100)
        print(con)
        print("Accuracy is {} %.".format(sum(np.diag(con)) / sum(sum(con)) * 100))
        train_mat_final = np.concatenate((loc_idx[:n_trn], tgt[:n_trn], x_train), axis=1)
        test_mat_final = np.concatenate((loc_idx[n_trn:], tgt[n_trn:], x_test, y_pred, prd_label), axis=1)
        train_Monte_file_nams = "./results/Radon_train" + str(mm) + ".txt"
        test_Monte_file_nams = "./results/Radon_test" + str(mm) + ".txt"
        np.savetxt(train_Monte_file_nams, train_mat_final)
        np.savetxt(test_Monte_file_nams, test_mat_final)
    print("="*100)
    print(con_sum)
    print("Average Accuracy is {} %.".format(sum(np.diag(con_sum)) / sum(sum(con_sum)) * 100))

data = pd.read_csv('./Data/Radon.txt', sep='  ', names=['G_label','Depth','T','pH','Eh','EC','DO','HCO3','Rn'])

# Average Performance of Random-undersampling
  y = data['Rn']
  num_data = data.drop(['Rn'], axis=1)
  def f(s):
      return (s-s.min())/(s.max()-s.min())
  x_normalization = num_data.apply(f, axis=0)
  x_normalization
  total = 0
  N = 1000
  for i in range(0,N):
      rus = RandomUnderSampler(random_state = i , replacement=False, sampling_strategy='majority')
      X_resampled, y_resampled = rus.fit_resample(x_normalization, y)
      X_resampled, y_resampled = rus.fit_resample(x_normalization, y)
      X_train, X_test, Y_train, Y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=20)
      rf.fit(X_train, Y_train)
      Pred = rf.predict(X_test)
      # print("Accuracy : {0:.3f}".format(accuracy_score(Y_test, Pred)))
      total = total + accuracy_score(Y_test, Pred)
  print(total/N)

# Concentration of Radon (Threshold = 100Bq/l)
dan_Rn = data[data['Rn'] == 1]
safe_Rn = data[data['Rn'] == 0]
dan_y = dan_Rn['Rn']
dan_x = dan_Rn.drop(['Rn'], axis=1)
safe_y = safe_Rn['Rn']
safe_x = safe_Rn.drop(['Rn'], axis=1)

# Noramalization (0 ~ 1) of data
def f(s):
    return (s-s.min())/(s.max()-s.min())
dan_x = dan_x.apply(f, axis=0)
safe_x = safe_x.apply(f, axis=0)


###### Over-Sampling ##### Making cross validation dataset #####
X_train, safe_x1, Y_train, safe_y1 = train_test_split(safe_x, safe_y, test_size=ceil(len(safe_Rn)/5), random_state=20)
X_train, safe_x2, Y_train, safe_y2 = train_test_split(X_train, Y_train, test_size=ceil(len(safe_Rn)/5), random_state=20)
X_train, safe_x3, Y_train, safe_y3 = train_test_split(X_train, Y_train, test_size=ceil(len(safe_Rn)/5), random_state=20)
X_train, safe_x4, Y_train, safe_y4 = train_test_split(X_train, Y_train, test_size=ceil(len(safe_Rn)/5), random_state=20)
dan_x5 = X_train
dan_y5 = Y_train

# Making Stratified cross validation dataset
# SCV1
Test_x1 = pd.concat([safe_x1, dan_x1])
Test_y1 = pd.concat([safe_y1, dan_y1])
Train_x1 = pd.concat([safe_x2, safe_x3, safe_x4, safe_x5, dan_x2, dan_x3, dan_x4, dan_x5])
Train_y1 = pd.concat([safe_y2, safe_y3, safe_y4, safe_y5, dan_y2, dan_y3, dan_y4, dan_y5])

# SCV2
Test_x2 = pd.concat([safe_x2, dan_x2])
Test_y2 = pd.concat([safe_y2, dan_y2])
Train_x2 = pd.concat([safe_x1, safe_x3, safe_x4, safe_x5, dan_x1, dan_x3, dan_x4, dan_x5])
Train_y2 = pd.concat([safe_y1, safe_y3, safe_y4, safe_y5, dan_y1, dan_y3, dan_y4, dan_y5])

# SCV3
Test_x3 = pd.concat([safe_x3, dan_x3])
Test_y3 = pd.concat([safe_y3, dan_y3])
Train_x3 = pd.concat([safe_x1, safe_x2, safe_x4, safe_x5, dan_x1, dan_x2, dan_x4, dan_x5])
Train_y3 = pd.concat([safe_y1, safe_y2, safe_y4, safe_y5, dan_y1, dan_y2, dan_y4, dan_y5])

# SCV4
Test_x4 = pd.concat([safe_x4, dan_x4])
Test_y4 = pd.concat([safe_y4, dan_y4])
Train_x4 = pd.concat([safe_x1, safe_x2, safe_x3, safe_x5, dan_x1, dan_x2, dan_x3, dan_x5])
Train_y4 = pd.concat([safe_y1, safe_y2, safe_y3, safe_y5, dan_y1, dan_y2, dan_y3, dan_y5])

# SCV5
Test_x5 = pd.concat([safe_x5, dan_x5])
Test_y5 = pd.concat([safe_y5, dan_y5])
Train_x5 = pd.concat([safe_x1, safe_x2, safe_x3, safe_x4, dan_x1, dan_x2, dan_x3, dan_x4])
Train_y5 = pd.concat([safe_y1, safe_y2, safe_y3, safe_y4, dan_y1, dan_y2, dan_y3, dan_y4])

# SMOTE(Oversampling)
method = SMOTE()
X_smote1, y_smote1 = method.fit_resample(Train_x1, Train_y1)

# Random forest training and testing (SCV)
rf = RandomForestClassifier(n_estimators=300, max_leaf_nodes=70, n_jobs=-1)
rf.fit(X_smote1, y_smote1)
Pred_SCV1 = rf.predict(Test_x1)

# Performance of RF
accuracy_rf = accuracy_score(Test_y1,Pred_SCV1)
f1_rf = f1_score(Test_y1,Pred_SCV1)
roc_score = roc_auc_score(Test_y1, Pred_SCV1)
print('Accuracy: ', accuracy_rf)
print('F1 score: ', f1_rf)
print('ROC AUC Score: ', roc_score)