import pickle
from scipy.stats import sem
import tensorflow
from sklearn.model_selection import  StratifiedKFold
import numpy as np
from numpy.random import seed
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers  import *
from sklearn.metrics import *
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model
from numpy.random import seed
from tensorflow.keras import backend as K
import time
seed(2022)
from tensorflow import set_random_seed
set_random_seed(2022)
class attention(Layer):
    def __init__(self, **kwargs):
        super(attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', shape=(input_shape[-1], 1),
                                 initializer='random_normal', trainable=True)
        self.b = self.add_weight(name='attention_bias', shape=(input_shape[1], 1),
                                 initializer='zeros', trainable=True)
        super(attention, self).build(input_shape)

    def call(self, x):
        e = K.tanh(K.dot(x, self.W) + self.b)
        e = K.squeeze(e, axis=-1)
        alpha = K.softmax(e)
        alpha = K.expand_dims(alpha, axis=-1)
        context = x * alpha
        context = K.sum(context, axis=1)
        return context
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
def weighted_bincrossentropy(true, pred, weight_zero=0.25, weight_one=1):
    bin_crossentropy = tensorflow.keras.backend.binary_crossentropy(true, pred)
    weights = true * weight_one + (1. - true) * weight_zero
    weighted_bin_crossentropy = weights * bin_crossentropy

    return tensorflow.keras.backend.mean(weighted_bin_crossentropy)
def mlp(input_shape):
    model = Sequential()
    model.add(Dense(input_shape,input_dim=input_shape,name='dense1'))
    model.add(BatchNormalization())
    model.add(Dense(512,name='dense2'))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(1e-4)
    model.compile(optimizer=adam, loss=weighted_bincrossentropy, metrics=['accuracy'])
    return model
def create_mlp_with_attention(input_shape):
    x=Input(shape=input_shape)
    attention_layer=attention()(x)
    dense2 = Dense(512)(attention_layer)
    batchN2 = BatchNormalization()(dense2)
    outputs=Dense(1,trainable=True,activation=Activation('sigmoid'))(batchN2)
    model=Model(x,outputs)
    adam = Adam(1e-4)
    model.compile(loss=weighted_bincrossentropy,optimizer=adam,metrics=['accuracy'])
    return model
def bilstm_with_mlp(input_shape):
    TIME_STEPS = 1
    INPUT_SIZE = input_shape
    model = Sequential()
    model.add(Bidirectional(LSTM(units=512,
                                 batch_input_shape=(None, TIME_STEPS, INPUT_SIZE),
                                 return_sequences=True,
                                 ), merge_mode='concat'))
    model.add(Flatten())
    model.add(BatchNormalization())
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Dense(1))
    model.add(Activation('sigmoid'))
    adam = Adam(1e-4)
    model.compile(optimizer=adam, loss=weighted_bincrossentropy, metrics=['accuracy'])
    return model
def get_data(path1, path2,feature_name):
    if(feature_name=='half_prot_t5_xl_uniref50_protein_embs'):
        with open(path1,"rb") as tf:
            feature_dict = pickle.load(tf)
        train_P = np.array([item for item in feature_dict.values()])
        tf.close()
        with open(path2,"rb") as tf:
            feature_dict = pickle.load(tf)
        train_N = np.array([item for item in feature_dict.values()])
        tf.close()
    else:
        data = np.load(path1)
        tmp = data.files
        for i in range(len(tmp)):
            train_P = data[tmp[i]]
        data = np.load(path2)
        tmp = data.files
        for i in range(len(tmp)):
            train_N = data[tmp[i]]
    train = np.vstack((train_P, train_N))
    return train
def get_label():
    train_label1=[1 for i in range(141)]
    train_label2=[0 for i in range(446)]
    train_label=train_label1+train_label2
    train_label=np.array(train_label)
    train_label=train_label.reshape((len(train_label),-1))
    test_label1 = [1 for i in range(34)]
    test_label2 = [0 for i in range(34)]
    test_label = test_label1 + test_label2
    test_label = np.array(test_label)
    test_label = test_label.reshape((len(test_label), -1))
    return train_label,test_label
def train_and_test(feature_name,model_name):
    if feature_name=='half_prot_t5_xl_uniref50_protein_embs':
        train_P_name = 'train_P_' + feature_name + '.pkl'
        train_N_name = 'train_N_' + feature_name + '.pkl'
        test_P_name = 'test_P_' + feature_name + '.pkl'
        test_N_name = 'test_N_' + feature_name + '.pkl'
    else:
        train_P_name='train_P_'+feature_name+'.npz'
        train_N_name='train_N_'+feature_name+'.npz'
        test_P_name='test_P_'+feature_name+'.npz'
        test_N_name='test_N_'+feature_name+'.npz'
    train_P_file_path='../pre_feature/train/'+train_P_name
    train_N_file_path='../pre_feature/train/'+train_N_name
    test_P_file_path='../pre_feature/test/'+test_P_name
    test_N_file_path='../pre_feature/test/'+test_N_name
    train=get_data(train_P_file_path,train_N_file_path,feature_name)
    test=get_data(test_P_file_path,test_N_file_path,feature_name)
    print(test.shape)
    input_shape = train.shape[1]
    train_label,test_label=get_label()
    indices=np.arange(len(train_label))
    np.random.shuffle(indices)
    train_label=train_label[indices]
    train=train[indices]
    y_train=train_label
    y_test=test_label
    if model_name=="mlp":
        pass
    else:
        train = train[:, np.newaxis]
        test = test[:, np.newaxis]
    mean_acc_score = []
    mean_auc_score = []
    mean_sn_score = []
    mean_sp_score = []
    mean_mcc_score = []
    mean_precision_scores = []
    mean_f1_scores = []
    mean_auprc_scores = []
    for x in range(5):
        print("*************************************************")
        print(x)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=x)
        acc_score = []
        auc_score = []
        sn_score = []
        sp_score = []
        mcc_score = []
        precision_scores = []
        f1_scores = []
        auprc_scores = []
        batch_size = 64
        nb_epoch = 60
        yy_pred_list = []
        true_pred_list=[]
        for i, (train_index, test_index) in enumerate(kf.split(train, y_train)):
            train_x1 = train[train_index]
            train_y1 = y_train[train_index]
            val_x = train[test_index]
            val_y = y_train[test_index]
            model_save_path='../model/'+feature_name+"_"+model_name+".h5"
            print(model_save_path)
            model_check = ModelCheckpoint(filepath=model_save_path ,
                                               monitor='val_acc', save_best_only=True)
            reduct_L_rate = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10)
            if model_name=='mlp':
                model = mlp(input_shape)
            elif model_name=='att':
                model=create_mlp_with_attention((1,input_shape))
                print("mlp_with_attention")
            elif model_name=='bilstm':
                model=bilstm_with_mlp(input_shape)

            train_x1 = np.array(train_x1, dtype=np.float32)
            train_y1 = np.array(train_y1, dtype=np.float32)
            test=np.array(test,dtype=np.float32)
            y_test=np.array(y_test,dtype=np.float32)
            val_x = np.array(val_x, dtype=np.float32)
            val_y = np.array(val_y, dtype=np.float32)
            history = model.fit(train_x1, train_y1, batch_size=batch_size, epochs=nb_epoch, validation_data=(val_x, val_y),
                                shuffle=True,callbacks=[model_check,reduct_L_rate]
            )
            model.summary()
            model = load_model(model_save_path,custom_objects={'attention':attention,'Activation':Activation,'weighted_bincrossentropy':weighted_bincrossentropy,})
            yy_pred = model.predict(val_x, batch_size=16, verbose=1)
            yy_pred_list.append(yy_pred)
            true_pred_list.append(val_y)
            acc,se,sp,mcc,auROC,precision,f1,AUPRC=eva(yy_pred, val_y)
            acc_score.append(acc)
            sn_score.append(se)
            sp_score.append(sp)
            mcc_score.append(mcc)
            auc_score.append(auROC)
            precision_scores.append(precision)
            f1_scores.append(f1)
            auprc_scores.append(AUPRC)
        path='./pred/5_fold/'+feature_name
        path2='./true'
        for a in range(len(true_pred_list)):
            tmp=true_pred_list[a]
            np.savez(path2+"/true_label"+str(x) +"_"+ str(a) + ".npz",tmp)
        for j in range(len(yy_pred_list)):
            tmp = yy_pred_list[j]
            np.savez(path+"/yy_pred_" +model_name+str(x) +"_"+ str(j) + ".npz", tmp)
        mean_acc = np.mean(acc_score)
        mean_auc = np.mean(auc_score)
        mean_sn = np.mean(sn_score)
        mean_sp = np.mean(sp_score)
        mean_mcc = np.mean(mcc_score)
        mean_precision=np.mean(precision_scores)
        mean_f1=np.mean(f1_scores)
        mean_auprc=np.mean(auprc_scores)
        mean_acc_score.append(mean_acc)
        mean_sp_score.append(mean_sp)
        mean_sn_score.append(mean_sn)
        mean_mcc_score.append(mean_mcc)
        mean_precision_scores.append(mean_precision)
        mean_f1_scores.append(mean_f1)
        mean_auprc_scores.append(mean_auprc)
        mean_auc_score.append(mean_auc)
    all_acc=np.mean(mean_acc_score)
    std_acc=sem(mean_acc_score)
    all_sp=np.mean(mean_sp_score)
    std_sp=sem(mean_sp_score)
    all_sn=np.mean(mean_sn_score)
    std_sn=sem(mean_sn_score)
    all_mcc=np.mean(mean_mcc_score)
    std_mcc=sem(mean_mcc_score)
    all_precision=np.mean(mean_precision_scores)
    std_precision=sem(mean_precision_scores)
    all_f1=np.mean(mean_f1_scores)
    std_f1=sem(mean_f1_scores)
    all_auprc=np.mean(mean_auprc_scores)
    std_auprc=sem(mean_auprc_scores)
    all_auc=np.mean(mean_auc_score)
    std_auc=sem(mean_auc_score)
    with open('result/5_fold_pre_independent_result.txt', 'a') as f:
        f.write(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        f.write("\n")
        f.write(feature_name)
        f.write(" ")
        f.write(model_name)
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
    parser.add_argument('-f',help='feature_name')
    parser.add_argument('-m', help='model_name')
    args=parser.parse_args()
    feature_name=args.f
    model_name=args.m
    train_and_test(feature_name,model_name)
