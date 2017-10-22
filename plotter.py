#Restrict to one gpu
import imp
try:
        imp.find_module('setGPU')
        import setGPU
except ImportError:
        found = False
#/////////////////////

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import keras.backend as K
import pylab as P
import pandas as pd
import numpy as np
import keras.callbacks
import glob

from sklearn.metrics import roc_auc_score,roc_curve,auc
from root_pandas import read_root

from keras.models import Model,load_model
from keras.layers import Input,Dense,Convolution1D,Flatten,Dropout,Activation

listOfFiles=glob.glob("/work/hajohajo/UnsupervisedJets/preprocessed_roots/*.root")
#read=['QG_ptD','QG_axis2','QG_mult','jet_eta','isPhysG','isPhysUD']

df = read_root(listOfFiles[-100:]) #,columns=read)
df['target'] = (df['isPhysG']==1)
df=df.drop(['isPhysG','isPhysUD'],axis=1)

test_y=df['target']
#df['ratio']=1.0*df[df['target']==1].shape[0]/df.shape[0]
test_x=df.drop('target',axis=1)


test_x=test_x.as_matrix()
test_y=test_y.as_matrix()

model=load_model('KERAS_model.h5')

pred_y=model.predict(test_x)

print ' - roc auc: ',round(roc_auc_score(test_y,pred_y),3)

print np.where(test_y==0)
gluons = pred_y[np.where(test_y==1)]
quarks = pred_y[np.where(test_y==0)]
#binning = np.linspace(0,1.0,20)
binning = np.arange(0.0,1.0,0.05)
print binning
plt.hist(gluons,bins=binning,alpha=0.8,label='Gluons',normed=1)
plt.hist(quarks,bins=binning,alpha=0.8,label='Quarks',normed=1)
plt.legend()
plt.title('Quark-Gluon classifier')
plt.xlabel('MVA output')
plt.ylabel('Jets')
plt.savefig('unsupClassif.pdf')

#ROC curve for plotting
fpr,tpr, thresholds  = roc_curve(test_y,pred_y)
roc_auc = auc(fpr, tpr)

plt.clf()
plt.plot(fpr,tpr,'b',label='Unsup. AUC = %0.2f'% roc_auc)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic")
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.savefig('roc_curve.pdf')
