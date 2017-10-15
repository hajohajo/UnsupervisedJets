#Restrict to one gpu
import imp
try:
        imp.find_module('setGPU')
        import setGPU
except ImportError:
        found = False
#/////////////////////
import tensorflow as tf
sess = tf.Session()
import matplotlib.pyplot as plt
import keras.backend as K
K.set_session(sess)
import pylab as P
import pandas as pd
import root_pandas
import numpy as np
import keras.callbacks
import glob
import math
from sklearn.metrics import roc_auc_score

loss_ = 'mean_squared_error'

#/////////////////TO BE MOVED INTO SEPARATE FILE FOR CLARITY
#ROC value to be printed out after epochs. Does not affect training
class ROC_value(keras.callbacks.Callback):
        def on_epoch_end(self, batch,logs={}):
                print ' - roc auc: ',round(roc_auc_score(test_y,self.model.predict(test_x)),3)

#Save losses etc. to a separate text file for plotting later
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
#        file=open("Losses_"+loss_+".txt","w+")
#        file.close()

        with open("Losses_"+loss_+".txt","a") as myfile:
                myfile.write('\n\n Model information:\n')
                self.model.summary(print_fn=lambda x: myfile.write(x + '\n'))
                myfile.write('\n\n Loss Accuracy        Val.loss        Val.Accuracy    ROC AUC\n')


    def on_epoch_end(self, batch, logs={}):

        string = str(round(logs.get('loss'),4))+"\t"+str(round(logs.get('val_loss'),4))+"\t"+str(round(logs.get('acc'),4))+"\t"+str(round(logs.get('val_acc'),4))+"\t"+str(round(roc_auc_score(test_y,self.model.predict(test_x)),3))+"\n"

        with open("Losses_"+loss_+".txt","a") as myfile:
                myfile.write(string)


def list_columns(obj, cols=4, columnwise=True, gap=4):
    """
    Print the given list in evenly-spaced columns.

    Parameters
    ----------
    obj : list
        The list to be printed.
    cols : int
        The number of columns in which the list should be printed.
    columnwise : bool, default=True
        If True, the items in the list will be printed column-wise.
        If False the items in the list will be printed row-wise.
    gap : int
        The number of spaces that should separate the longest column
        item/s from the next column. This is the effective spacing
        between columns based on the maximum len() of the list items.
    """

    sobj = [str(item) for item in obj]
    if cols > len(sobj): cols = len(sobj)
    max_len = max([len(item) for item in sobj])
    if columnwise: cols = int(math.ceil(float(len(sobj)) / float(cols)))
    plist = [sobj[i: i+cols] for i in range(0, len(sobj), cols)]
    if columnwise:
        if not len(plist[-1]) == cols:
            plist[-1].extend(['']*(len(sobj) - len(plist[-1])))
        plist = zip(*plist)
    printer = '\n'.join([
        ''.join([c.ljust(max_len + gap) for c in p])
        for p in plist])
    return printer



#/////////////////////////////////////////////

listOfFiles=glob.glob("/work/hajohajo/UnsupervisedJets/preprocessed_roots/*.root")


df = root_pandas.read_root(listOfFiles,'tree')
#gluons = df[(df.isPhysG == 1)]
#quarks = df[(df.isPhysUD == 1)]

targets = np.array(df.apply(lambda row: 1 if row.isPhysG == 1 else 0,axis=1))
df.drop(['isPhysG','isPhysUD'],axis=1,inplace=True)

train_x = df.sample(frac=0.9,random_state=7)
train_y = targets[train_x.index]
test_x = df.drop(train_x.index)
test_y = targets[test_x.index]

train_x = np.array(train_x.iloc[:])
test_x = np.array(test_x.iloc[:])


from keras.models import Model
from keras.layers import Input,Dense,Convolution1D,Flatten,Dropout,Activation
import keras.backend as K
from sklearn.utils import class_weight
from keras.layers.normalization import BatchNormalization

#Create a file to save info of the training
file=open("Losses_"+loss_+".txt","w+")
file.close()

with open("Losses_"+loss_+".txt","a") as myfile:
        myfile.write("\n\n Used variables:\n")
        myfile.write(list_columns(df.columns.values,cols=4))

#Defining the network topology
dropoutRate=0.1
a_inp = Input(shape=(train_x.shape[1],),name='ins')

a = Dense(500,activation='relu', kernel_initializer='normal')(a_inp)
#a = BatchNormalization()(a_inp)
a = Dropout(dropoutRate)(a)
a = Dense(250,activation='relu', kernel_initializer='normal')(a)
#a = BatchNormalization()(a)
a = Dropout(dropoutRate)(a)
a = Dense(120,activation='relu', kernel_initializer='normal')(a)
#a = BatchNormalization()(a)
a = Dropout(dropoutRate)(a)
a = Dense(20,activation='relu', kernel_initializer='normal')(a)
#a = BatchNormalization()(a)
a_out = Dense(1, activation='sigmoid', kernel_initializer='normal',name='outs')(a)

model=Model(inputs=a_inp,outputs=a_out)

from keras import optimizers
adam=optimizers.Adam() #lr=1.0)
model.compile(loss=loss_,optimizer=adam,metrics=['acc'])


cb=ROC_value()
loss=LossHistory()
check=keras.callbacks.ModelCheckpoint('KERAS_best_model_'+loss_+'.h5',monitor='val_loss',save_best_only=True)
class_weight = class_weight.compute_class_weight('balanced', np.unique(train_y[:]),train_y[:])
#class_weight = class_weight.compute_class_weight('balanced',np.unique(algos),algos[:])
Nepoch=100
batchS=512
model.fit(train_x,train_y,
        epochs=Nepoch,
        batch_size=batchS,
        class_weight=class_weight,
        callbacks=[cb,loss,check],
        validation_split=0.1,
        shuffle=True)

model.save('my_model_'+loss_+'_Adam.h5')

