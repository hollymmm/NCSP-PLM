import pickle
import pandas as pd
import tensorflow
from sklearn.model_selection import train_test_split, GridSearchCV
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

    train, val, y_train, y_val = train_test_split(train, y_train,test_size=0.25,random_state=2022)
    batch_size = 64
    nb_epoch =60
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
    train=np.array(train,dtype=np.float32)
    y_train=np.array(y_train,dtype=np.float32)
    test=np.array(test,dtype=np.float32)
    y_test=np.array(y_test,dtype=np.float32)
    history = model.fit(train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(val, y_val),
                        shuffle=True,callbacks=[model_check,reduct_L_rate]
    )
    model.summary()
    model = load_model(model_save_path,custom_objects={'attention':attention,'Activation':Activation,'weighted_bincrossentropy':weighted_bincrossentropy,})

    yy_pred = model.predict(test, batch_size=16, verbose=1)
    true_values = y_test
    acc,se,sp,mcc,auROC,precision,f1,AUPRC=eva(yy_pred, true_values)
    print("sensitivity/Recall:", se)
    print("specificity:", sp)
    print("precision:", precision)
    print("accuracy:", acc)
    print("Mcc:", mcc)
    print("f1:", f1)
    print("AUROC:", auROC)
    print("AUPRC", AUPRC)
    with open('result/singel_pre_independent_result.txt', 'a') as f:
        f.write(str(time.strftime('%Y-%m-%d %H:%M:%S',time.localtime(time.time()))))
        f.write("\n")
        f.write(feature_name)
        f.write(" ")
        f.write(model_name)
        f.write('\n')
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
        f.write("\n")
    np.savez("./pred/"+feature_name+'_'+model_name+ ".npz",yy_pred)
if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('-f',help='feature_name')
    parser.add_argument('-m', help='model_name')
    args=parser.parse_args()
    feature_name=args.f
    model_name=args.m
    train_and_test(feature_name,model_name)
