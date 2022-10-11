import numpy as np
from sklearn.metrics import *

def read(fath_name):
    data=np.load(fath_name)
    tmp = data.files
    for i in range(len(tmp)):
        pred = data[tmp[i]]
    return pred
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
    precision=precision_score(true_values,y_labels)
    f1=f1_score(true_values,y_labels)
    precision2, recall, thresholds = precision_recall_curve(true_values, y_labels)
    AUPRC = auc(recall, precision2)
    return acc, se, sp, mcc, AUC,precision,f1,AUPRC

def To_vote(weight):
    data_list=[]
    fath_name=['./pred/esm_att.npz','./pred/proSE_bilstm.npz','./pred/proteinbertglobal_att.npz','./pred/half_prot_t5_xl_uniref50_protein_embs_mlp.npz','./pred/unirep_att.npz']
    for i in range(len(fath_name)):
        data=read(fath_name[i])
        data=data.reshape(-1)
        data_list.append(data)
    data_list=np.array(data_list)
    mean=np.average(data_list, axis=0,weights=weight)
    data2 = np.load('./pred/true_values'+'.npz')
    tmp = data2.files
    for i in range(len(tmp)):
        true_label0 = data2[tmp[i]]
    acc,se,sp,mcc,auROC,precision,f1,AUPRC=eva(mean,true_label0)
    with open('result/vote_result.txt', 'w') as f:
        f.write(str(se))
        f.write(" ")
        f.write(str(sp))
        f.write(" ")
        f.write(str(precision))
        f.write(" ")
        f.write(str(acc))
        f.write(" ")
        f.write(str(mcc))
        f.write(" ")
        f.write(str(f1))
        f.write(" ")
        f.write(str(auROC))
        f.write(" ")
        f.write(str(AUPRC))
    print("sensitivity/Recall:",se)
    print("specificity:",sp)
    print("precision:",precision)
    print("accuracy:",acc)
    print("Mcc:",mcc)
    print("f1:",f1)
    print("AUROC:",auROC)
    print("AUPRC",AUPRC)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-w',help='weight',nargs=5,type=int)
    args=parser.parse_args()
    weight=args.w
    To_vote(weight)

