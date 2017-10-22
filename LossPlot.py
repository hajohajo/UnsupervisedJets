import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np

f=open("Loss_to_plot.txt","r") #_BENCHMARK.txt","r")
lines=f.readlines()
train_loss=[]
val_loss=[]
for x in lines:
    train_loss.append(x.split()[0])
    val_loss.append(x.split()[1])
f.close()

xrange=range(1,len(train_loss)+1)

plt.plot(xrange,train_loss,label='Training loss')
plt.plot(xrange,val_loss,label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Binary crossentropy')
plt.title("Training and validation loss")
plt.legend()
plt.savefig("Losses.pdf")
