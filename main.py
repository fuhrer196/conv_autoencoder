from __future__ import division, print_function, absolute_import
import argparse

parser = argparse.ArgumentParser(description='Generate an autoencoder for given data')
parser.add_argument('-e', action="store", dest="training_epochs", type=int, default=1500, help="Number of training epochs (default 1500)")
parser.add_argument('-r', action="store", dest="alpha", type=float, default=0.0005, help="Regularisation coefficient (default 0.0005)")
parser.add_argument('-l', action="store", dest="learning_rate", type=float, default=0.01, help="Learning rate (default 0.01)")
parser.add_argument('-n', action="store",required=True, dest="res_n", help="label for output matrix (result_[n].mat)")
parser.add_argument('-b', action="store",required=False, dest="batch_size", type=int, default=64, help="minibatch size (defaults to 64)")

parser.add_argument('--length', action="store", dest="length", type=float, default=0.000001, help="Coefficient of length penalty(default 0.0005)")
parser.add_argument('--area', action="store", dest="area", type=float, default=0.001, help="Coefficient of area penalty(default 0.0005)")
parser.add_argument('--roughness', action="store", dest="roughness", type=float, default=5.0, help="Coefficient of roughness penalty(default 0.0005)")

parser.add_argument('--restart', dest='restart', default=False, action='store_true')
parser.add_argument('--log', dest='log', default=False, action='store_true')
parser.add_argument('--data', action="store", dest="inp_filename", required=False, default="inp.mat", help="Filename of .mat datafile. Defaults to 'inp.mat'. Format: {x:xdata, y:ydata}")
parser.parse_args()

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb
import os
import shutil
from data_handler import DataWrapper
import tensorflow as tf
from timeit import default_timer as timer
import csv

p = 4 #numchannels
w = 5 #window

alpha           = parser.parse_args().alpha
learning_rate   = parser.parse_args().learning_rate
training_epochs = parser.parse_args().training_epochs
save_to         = "result" + parser.parse_args().res_n
regularizer = tf.contrib.layers.l2_regularizer(alpha)
batch_size = parser.parse_args().batch_size
restart = parser.parse_args().restart
log = parser.parse_args().log
data = DataWrapper(filename=parser.parse_args().inp_filename, batch_size=batch_size)


if restart:
    try:
        os.remove('costs'+parser.parse_args().res_n+'.csv')
        shutil.rmtree('timelySave'+parser.parse_args().res_n)
    except:
        print("One or more of costs[n].csv, result[n].mat, timelySave[n] missing")

X = tf.placeholder("float32", [None, 64, 1])
Y = tf.placeholder("float32", [None, 64, 1])
conv1 = []; pool1 = []; conv2 = []; pool2 = []; fc1=[];
for i, Dim in enumerate([X,Y]):
    conv1.append(tf.layers.conv1d(Dim, p, w, kernel_regularizer=regularizer, data_format="channels_last"))                                 #(?,4,60 (+4))
    conv1[i] = tf.maximum(conv1[i], -0.01*conv1[i])
    pool1.append(tf.layers.max_pooling1d(conv1[i],2,2, data_format="channels_last"))                   #(?,4,30 (+2))
    conv2.append(tf.layers.conv1d(pool1[i],int(p*(p-1)/2),w, kernel_regularizer=regularizer, data_format="channels_last"))                 #(?,32,6)
    conv2[i] = tf.maximum(conv2[i], -0.01*conv2[i])
    pool2.append(tf.layers.max_pooling1d(conv2[i],4,4, data_format="channels_last"))                   #(?,8,6)
    #for channel in X[]:
    fc1.append(tf.layers.dense(tf.transpose(pool2[i], perm=[0,2,1]),8, kernel_regularizer=regularizer))                                    #(?,6,8)
    fc1[i] = tf.maximum(fc1[i], -0.01*fc1[i])
    #fc1.unroll()

conc        = tf.concat([fc1[0],fc1[1]], -1)
reshaped    = tf.reshape( conc , [-1,p*(p-1)*8])
enc = tf.layers.dense( reshaped, 20, kernel_regularizer=regularizer)
enc = tf.maximum(enc, -0.01 * enc) #(b,20)

#DECODER
d_fc2 = tf.split(tf.layers.dense(enc, int(8*p*(p-1)), kernel_regularizer=regularizer), num_or_size_splits=2, axis=1)#[(b,48),(b,48)]
d_fc2 = [tf.reshape(tf.maximum(i, -0.01 * i),[-1, int(p*(p-1)/2),8]) for i in d_fc2] #[(b,6,8),(b,6,8)]

d_fc1 = [tf.layers.dense(i, 8, kernel_regularizer=regularizer) for i in d_fc2] 
d_fc1 = [tf.maximum(i, -0.01 * i) for i in d_fc1] #[(b,6,8),(b,6,8)]

d_pool2 = [tf.reshape(tf.transpose(tf.concat([[i],[i],[i],[i]],axis=0),[1,2,3,0]),[-1,int(p*(p-1)/2),32]) for i in d_fc1] #[(b,6,32),...]
d_conv2 = [tf.layers.conv1d(i,p,w,kernel_regularizer=regularizer, data_format="channels_first", padding='same') for i in d_pool2]  #[(b,6,32)]
d_conv2 = [tf.maximum(i, -0.01 * i) for i in d_conv2]
d_pool1 = [tf.reshape(tf.transpose(tf.concat([[i],[i]],axis=0),[1,2,3,0]),[-1,p,64]) for i in d_conv2] #[(b,6,64)]
d_conv1 = [tf.layers.conv1d(i,1,w,kernel_regularizer=regularizer, data_format="channels_first", padding='same') for i in d_pool1] #[b,1,64]
d_conv1 = [tf.maximum(i, -0.01 * i) for i in d_conv1]

reg_term = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))

x = tf.reshape(d_conv1[0], [-1, 64])
y = tf.reshape(d_conv1[1], [-1, 64])

IK       = np.fft.fftfreq(64)*1j
IK       = IK.astype(np.dtype('complex64'))
dbydx    = tf.real(tf.ifft(tf.multiply(IK,tf.fft(tf.complex(x, 0.0)))))
dbydy    = tf.real(tf.ifft(tf.multiply(IK,tf.fft(tf.complex(y, 0.0)))))
length   = tf.reduce_sum(tf.sqrt(tf.add(tf.square(dbydx),tf.square(dbydy))), 1)
area     = tf.reduce_sum(tf.add(tf.multiply(x,dbydy),-1*tf.multiply(y,dbydx)), 1)

r_x = tf.reshape(X, [-1, 64])
r_y = tf.reshape(Y, [-1, 64])
r_dbydx    = tf.real(tf.ifft(tf.multiply(IK,tf.fft(tf.complex(r_x, 0.0)))))
r_dbydy    = tf.real(tf.ifft(tf.multiply(IK,tf.fft(tf.complex(r_y, 0.0)))))
r_length   = tf.reduce_sum(tf.sqrt(tf.add(tf.square(r_dbydx),tf.square(r_dbydy))), 1)
r_area     = tf.reduce_sum(tf.add(tf.multiply(r_x,r_dbydy),-1*tf.multiply(r_y,r_dbydx)), 1)

c1       = tf.add_n([tf.reduce_sum(tf.pow(y - r_y, 2)), tf.reduce_sum(tf.pow(x - r_x, 2))])
c2       = parser.parse_args().roughness*tf.add_n([tf.reduce_sum(tf.square(dbydx)), tf.reduce_sum(tf.square(dbydy))])
c3       = parser.parse_args().length*tf.reduce_sum(tf.pow(length-r_length,2))
c4       = parser.parse_args().area*tf.reduce_sum(tf.pow((area-r_area)/r_area,2)) #1e-2

cost     = tf.add_n([c1 , c2, c3, c4, reg_term])

global_step = tf.Variable(0, name='global_step', trainable=False)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

saver = tf.train.Saver()
class saveHook(tf.train.SessionRunHook):
    def begin(self):
        self.train_time = timer()
    def after_run(self, run_context, fuckit):
        sess = run_context.session
        batch = data.getBatch()
        if global_step.eval(session=sess)%5000 == 0:
            y_pred, x_pred = sess.run([y, x], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
            tosave = {  'x_act':batch["x"],
                        'y_act':batch["y"],
                        'x_pred':x_pred,
                        'y_pred':y_pred,
                        'learning_rate': learning_rate,
                        'training_epochs':training_epochs,
                        'alpha_reg': alpha,
                        'costs':costs,
                        'regs':regs,}
            sio.savemat("live" + str(parser.parse_args().res_n) + ".mat",tosave)
    def end(self, sess):
        self.train_time = timer() - self.train_time
        batch = data.getBatch()
        y_pred, x_pred, cst, reg, c_l2, c_area, c_ln,c_roughness = sess.run([y, x, cost, reg_term,c1,c4,c3,c2], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
        tosave = {  'x_act':batch["x"],
                    'y_act':batch["y"],
                    'x_pred':x_pred,
                    'y_pred':y_pred,
                    'learning_rate': learning_rate,
                    'training_epochs':training_epochs,
                    'alpha_reg': alpha,
                    'regs':regs,
                    'train_time': self.train_time,
                    'l2':c1s,
                    'len':c3s,
                    'area':c4s,
                    'roughness':c2s,
                    'regs':regs,
                    'costs':costs}
        sio.savemat(save_to+".mat",tosave)

        batch = data.getValidationData()
        val_size = len(batch["x"])
        start = timer()
        y_pred, x_pred, cst_val, reg, c_l2, c_area, c_ln,c_roughness = sess.run([y, x, cost, reg_term,c1,c4,c3,c2], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
        val_time = timer() - start

        tosave = {  'x_act':batch["x"],
                    'y_act':batch["y"],
                    'x_pred':x_pred,
                    'y_pred':y_pred,
                    'total_cost':cst_val,
                    'l2':c_l2,
                    'len':c_ln,
                    'area':c_area,
                    'roughness':c_roughness,
                    'val_time': val_time,
                    'reg':reg,
                    }
        sio.savemat("validation" + str(parser.parse_args().res_n) + ".mat",tosave)

        batch = data.getTestData()
        y_pred, x_pred, cst, reg, c_l2, c_area, c_ln,c_roughness = sess.run([y, x, cost, reg_term,c1,c4,c3,c2], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
        tosave = {  'x_act':batch["x"],
                    'y_act':batch["y"],
                    'x_pred':x_pred,
                    'y_pred':y_pred,
                    'total_cost':cst,
                    'l2':c_l2,
                    'len':c_ln,
                    'area':c_area,
                    'roughness':c_roughness,
                    'reg':reg,
                    }
        sio.savemat("test" + str(parser.parse_args().res_n) + ".mat",tosave)
        if log:
            if not os.path.exists('log.csv'):
                with open('log.csv', 'wb') as logfile:
                    logger = csv.writer(logfile, delimiter=',')
                    logger.writerow(['Run no.'         , 'Number of Epochs'   , 'Batch_size'    , 'Regularisation ceoff' ,
                                     'Learning Rate'   , 'Length coeff'       , 'Area coeff'    , 'Roughness coeff'      ,
                                     'Validation time' , 'Cost on validation' , 'Cost on train' , 'Train time'           ,
                                     'L2 error on training', 'L2 error on validation'])
            with open('log.csv', 'ab') as logfile:
                logger = csv.writer(logfile, delimiter=',')
                logger.writerow([parser.parse_args().res_n , sess.run(global_step)      , batch_size               , alpha                         ,
                                  learning_rate            , parser.parse_args().length , parser.parse_args().area , parser.parse_args().roughness ,
                                  val_time                 , cst_val/val_size           , costs[-1]/batch_size     , self.train_time               ,
                                  c1s[-1]/batch_size       , c_l2/val_size])

hooks=[tf.train.StopAtStepHook(num_steps=training_epochs), saveHook()]


costs = []
c1s = []
c2s = []
c3s = []
c4s = []
regs = []

f = open('costs'+str(parser.parse_args().res_n)+'.csv', 'a')

with tf.train.MonitoredTrainingSession(checkpoint_dir="./timelySave"+str(parser.parse_args().res_n)+"/",
                                       hooks=hooks) as mon_sess:

    tf.set_random_seed(1)
    batch = data.getBatch()
    while not mon_sess.should_stop():
        _, cst,reg, gs,lc1,lc2,lc3,lc4 =  mon_sess.run([optimizer, cost, reg_term, global_step, c1,c2,c3,c4], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
        costs.append(cst)
        c1s.append(lc1)
        c2s.append(lc2)
        c3s.append(lc3)
        c4s.append(lc4)
        regs.append(reg)
        f.write(str(cst)+','+str(lc1)+','+str(lc2)+','+str(lc3)+','+str(lc4)+','+str(reg)+"\n")
print("Optimization Finished!") 
f.close()


