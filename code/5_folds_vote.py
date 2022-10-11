from scipy.stats import sem
import numpy as np
from sklearn.metrics import *
from numpy.random import seed
import time
seed(2022)
from tensorflow import set_random_seed
set_random_seed(2022)
import os


def get_true_label():
    path = 'true'
    file_name_list2 = os.listdir(path)
    data_list2 = []
    for i in range(len(file_name_list2)):
        read_path = path + '/' + file_name_list2[i]
        data = np.load(read_path)
        tmp = data.files
        for j in range(len(tmp)):
            data_list2.append(data[tmp[j]])
    return data_list2


def get_model_pred(feature_name, model_name):
    path = './pred/5_fold/' + feature_name
    path_list = [os.path.join(path, i) for i in os.listdir(path) if i.find(model_name) != -1]

    data_list = []
    for i in range(len(path_list)):
        read_path = path_list[i]
        data = np.load(read_path)
        print(read_path)
        tmp = data.files
        for j in range(len(tmp)):
            data_list.append(data[tmp[j]])
    return data_list
def eva(yy_pred, true_values):
    y_pred = yy_pred
    y_scores = np.array(y_pred)
    AUC = roc_auc_score(true_values, y_scores)
    y_labels = []
    y_scores = y_scores.reshape((len(y_scores), -1))
    for i in range(len(y_scores)):
        if (y_scores[i] >= 0.5):
            y_labels.append(1)
        else:
            y_labels.append(0)
    fpr, tpr, thresholds = roc_curve(true_values, y_labels)
    acc = accuracy_score(true_values, y_labels)
    se = recall_score(true_values, y_labels)
    sp = 1 - fpr[1]
    mcc = matthews_corrcoef(true_values, y_labels)
    precision = precision_score(true_values, y_labels)
    f1 = f1_score(true_values, y_labels)
    precision2, recall, thresholds = precision_recall_curve(true_values, y_labels)
    AUPRC = auc(recall, precision2)
    return acc, se, sp, mcc, AUC, precision, f1, AUPRC
def To_vote(weight):
    true_label=get_true_label()
    esm_pred=get_model_pred('esm','att')
    ProSE_pred=get_model_pred('ProSE','bilstm')
    global_pred=get_model_pred('proteinbertglobal','att')
    t5_pred=get_model_pred('half_prot_t5_xl_uniref50_protein_embs','mlp')
    unirep_pred=get_model_pred('unirep','att')
    acc_score=[]
    se_score=[]
    sp_score=[]
    mcc_score=[]
    auROC_score=[]
    precision_scores=[]
    f1_scores=[]
    AUPRC_score=[]
    for i in range(5):
        fold1_1=np.concatenate((esm_pred[i],ProSE_pred[i],global_pred[i],t5_pred[i],unirep_pred[i]),axis=1)
        fold2_1=np.concatenate((esm_pred[i+5],ProSE_pred[i+5],global_pred[i+5],t5_pred[i+5],unirep_pred[i+5]),axis=1)
        fold3_1=np.concatenate((esm_pred[i+10],ProSE_pred[i+10],global_pred[i+10],t5_pred[i+10],unirep_pred[i+10]),axis=1)
        fold4_1=np.concatenate((esm_pred[i+15],ProSE_pred[i+15],global_pred[i+15],t5_pred[i+15],unirep_pred[i+15]),axis=1)
        fold5_1=np.concatenate((esm_pred[i+20],ProSE_pred[i+20],global_pred[i+20],t5_pred[i+20],unirep_pred[i+20]),axis=1)
        five_fold1=np.concatenate((fold1_1,fold2_1,fold3_1,fold4_1,fold5_1),axis=0)
        mean=np.average(five_fold1, axis=1,weights=weight)
        true_label1=np.concatenate((true_label[i],true_label[i+5],true_label[i+10],true_label[i+15],true_label[i+20]))
        acc,se,sp,mcc,auROC,precision,f1,AUPRC=eva(mean,true_label1)
        print("sensitivity/Recall:", se)
        print("specificity:", sp)
        print("precision:", precision)
        print("accuracy:", acc)
        print("Mcc:", mcc)
        print("f1:", f1)
        print("AUROC:", auROC)
        print("AUPRC", AUPRC)
        print("###########################################")
        acc_score.append(acc)
        se_score .append(se)
        sp_score .append(sp)
        mcc_score .append(mcc)
        auROC_score .append(auROC)
        precision_scores .append(precision)
        f1_scores .append(f1)
        AUPRC_score .append(AUPRC)
    all_acc=np.mean(acc_score)
    std_acc=sem(acc_score)
    all_sp=np.mean(sp_score)
    std_sp=sem(sp_score)
    all_sn=np.mean(se_score)
    std_sn=sem(se_score)
    all_mcc=np.mean(mcc_score)
    std_mcc=sem(mcc_score)
    all_precision=np.mean(precision_scores)
    std_precision=sem(precision_scores)
    all_f1=np.mean(f1_scores)
    std_f1=sem(f1_scores)
    all_auprc=np.mean(AUPRC_score)
    std_auprc=sem(AUPRC_score)
    all_auc=np.mean(auROC_score)
    std_auc=sem(auROC_score)
    with open('result/vote_result.txt', 'a') as f:
        f.write(str(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))))
        f.write(" ")
        f.write('\n')
        f.write(str(all_sn))
        f.write(" ")
        f.write(str(all_sp))
        f.write(" ")
        f.write(str(all_precision))
        f.write(" ")
        f.write(str(all_acc))
        f.write(" ")
        f.write(str(all_mcc))
        f.write(" ")
        f.write(str(all_f1))
        f.write(" ")
        f.write(str(all_auc))
        f.write(" ")
        f.write(str(all_auprc))
        f.write("\n")
        f.write(str(std_sn))
        f.write(" ")
        f.write(str(std_sp))
        f.write(" ")
        f.write(str(std_precision))
        f.write(" ")
        f.write(str(std_acc))
        f.write(" ")
        f.write(str(std_mcc))
        f.write(" ")
        f.write(str(std_f1))
        f.write(" ")
        f.write(str(std_auc))
        f.write(" ")
        f.write(str(std_auprc))
        f.write("\n")
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-w',help='weight',nargs=5,type=int)
    args=parser.parse_args()
    weight=args.w
    To_vote(weight)
