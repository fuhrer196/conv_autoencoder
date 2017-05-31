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

parser.add_argument('--data', action="store", dest="inp_filename", required=False, default="inp.mat", help="Filename of .mat datafile. Defaults to 'inp.mat'. Format: {x:xdata, y:ydata}")
parser.parse_args()

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb
import os
from data_handler import DataWrapper
import tensorflow as tf

p = 4 #numchannels
w = 5 #window

alpha           = parser.parse_args().alpha
learning_rate   = parser.parse_args().learning_rate
training_epochs = parser.parse_args().training_epochs
save_to         = "result" + parser.parse_args().res_n
regularizer = tf.contrib.layers.l2_regularizer(alpha)
batch_size = parser.parse_args().batch_size
data = DataWrapper(filename=parser.parse_args().inp_filename, batch_size=batch_size)


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
d_pool1 = [tf.reshape(tf.transpose(tf.concat([[i],[i]],axis=0),[1,2,3,0]),[-1,p,64]) for i in d_conv2] #[(b,6,64)]
d_conv1 = [tf.layers.conv1d(i,1,w,kernel_regularizer=regularizer, data_format="channels_first", padding='same') for i in d_pool1] #[b,1,64]

# dec_1 = [tf.reshape(i, [-1, 8, 1, int(p*(p-1)/2)]) for i in fc_deconv2] # height 8, width 1, channel p*(p-1)/2. NHWC

# deconv1 = [0, 0]
# deconv2 = [0, 0]
# filter_deconv2 = [0, 0]
# filter_deconv1 = [0, 0]

# for i in xrange(2):
#     filter_deconv1[i] = tf.Variable(tf.random_normal([w, 1, p, int(p*(p-1)/2)], stddev=0.5))
#     deconv1[i] = tf.nn.conv2d_transpose(dec_1[i], filter_deconv1[i], output_shape=[batch_size,32, 1, p], strides=[1, 4, 4, 1])
#     deconv1[i] = tf.maximum(deconv1[i], -0.01 * deconv1[i])
#     filter_deconv2[i] = tf.Variable(tf.random_normal([w, 1, 1, p] , stddev=0.5))
#     deconv2[i] = tf.nn.conv2d_transpose(deconv1[i], filter_deconv2[i], output_shape=[batch_size,64, 1, 1], strides=[1, 2, 2, 1])
#     deconv2[i] = tf.maximum(deconv2[i], -0.01 * deconv2[i])

#     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,regularizer(filter_deconv1[i]))
#     tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,regularizer(filter_deconv2[i]))

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

c1       = tf.add_n([tf.reduce_sum(tf.pow(y - r_y, 2)), tf.reduce_sum(tf.pow(x - r_x, 2)), reg_term])
c2       = parser.parse_args().roughness*tf.add_n([tf.reduce_sum(tf.square(dbydx)), tf.reduce_sum(tf.square(dbydy))])
c3       = parser.parse_args().length*tf.reduce_sum(tf.pow(length-r_length,2))
c4       = parser.parse_args().area*tf.reduce_sum(tf.pow((area-r_area)/r_area,2)) #1e-2

cost     = tf.add_n([c1 , c2, c3, c4])

global_step = tf.Variable(0, name='global_step', trainable=False)

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

saver = tf.train.Saver()
class saveHook(tf.train.SessionRunHook):
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
                        'costs':costs}
            sio.savemat("live" + str(parser.parse_args().res_n) + ".mat",tosave)
    def end(self, sess):
        batch = data.getBatch()
        y_pred, x_pred = sess.run([y, x], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
        tosave = {  'x_act':batch["x"],
                    'y_act':batch["y"],
                    'x_pred':x_pred,
                    'y_pred':y_pred,
                    'learning_rate': learning_rate,
                    'training_epochs':training_epochs,
                    'alpha_reg': alpha,
                    'costs':costs}
        sio.savemat(save_to+".mat",tosave)

        batch = data.getValidationData()
        y_pred, x_pred, cst, reg = sess.run([y, x, cost, reg_term], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
        tosave = {  'x_act':batch["x"],
                    'y_act':batch["y"],
                    'x_pred':x_pred,
                    'y_pred':y_pred,
                    'cost':cst,
                    'reg':reg_term,
                    }
        sio.savemat("validation" + str(parser.parse_args().res_n) + ".mat",tosave)

        batch = data.getTestData()
        y_pred, x_pred, cst = sess.run([y, x, cost], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
        tosave = {  'x_act':batch["x"],
                    'y_act':batch["y"],
                    'x_pred':x_pred,
                    'y_pred':y_pred,
                    'cost':cst,}
        sio.savemat("test" + str(parser.parse_args().res_n) + ".mat",tosave)

hooks=[tf.train.StopAtStepHook(num_steps=training_epochs), saveHook()]


costs = []

f = open('costs'+str(parser.parse_args().res_n)+'.csv', 'a')

with tf.train.MonitoredTrainingSession(checkpoint_dir="./timelySave"+str(parser.parse_args().res_n)+"/",
                                       hooks=hooks) as mon_sess:

    batch = data.getBatch()
    #enc,dfc1 =  mon_sess.run([enc, d_fc1], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
    y_pred, x_pred = mon_sess.run([y, x], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
    while not mon_sess.should_stop():
        _, cst, gs =  mon_sess.run([optimizer, cost, global_step], feed_dict={X: np.transpose([batch["x"]],(1,2,0)),Y: np.transpose([batch["y"]],(1,2,0)) })
        costs.append(cst)
        f.write(str(cst)+"\n")
print("Optimization Finished!") 
f.close()
