from __future__ import division, print_function, absolute_import
import argparse

parser = argparse.ArgumentParser(description='Generate an autoencoder for given data')
parser.add_argument('-e', action="store", dest="training_epochs", type=int, default=1500, help="Number of training epochs (default 1500)")
parser.add_argument('-r', action="store", dest="alpha", type=float, default=0.0005, help="Regularisation coefficient (default 0.0005)")
parser.add_argument('-l', action="store", dest="learning_rate", type=float, default=0.01, help="Learning rate (default 0.01)")
parser.add_argument('-n', action="store",required=True, dest="res_n", help="label for output matrix (result_[n].mat)")

parser.add_argument('--length', action="store", dest="length", type=float, default=0.000001, help="Coefficient of length penalty(default 0.0005)")
parser.add_argument('--area', action="store", dest="area", type=float, default=0.001, help="Coefficient of area penalty(default 0.0005)")
parser.add_argument('--roughness', action="store", dest="roughness", type=float, default=5.0, help="Coefficient of roughness penalty(default 0.0005)")
parser.parse_args()

import sys
import time
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import pdb
import os

import tensorflow as tf

p = 4 #numchannels
w = 5 #window

alpha           = parser.parse_args().alpha
learning_rate   = parser.parse_args().learning_rate
training_epochs = parser.parse_args().training_epochs
save_to         = "result" + parser.parse_args().res_n

X = tf.placeholder("floa32", [None, 64])
Y = tf.placeholder("floa32", [None, 64])

for i, Dim in enumerate([X,Y]):
    conv1[i]  = tf.layers.conv1d(Dim, p,w,data_format="channels_last", padding='SAME')
    pool1[i]  = tf.layers.average_pooling1d(conv1, 2, 2)
    conv2[i]  = tf.layers.conv1d(pool1, p*(p-1)/2, w, padding='SAME')
    pool2[i]  = tf.layers.average_pooling1d(conv2, 4, 4)

    #for channel in X[]:
    fc1[i]    = tf.layers.dense(pool2,8)
    #fc1.unroll()

enc = tf.layers.dense(tf.concat([fc1[0],fc1[1]]),16)

#DECODER
fc_deconv = tf.split(tf.layers.dense(enc, int(16*p*(p-1)/2)), num_or_size_splits=2, axis=1)
fc_deconv2 = [tf.layers.dense(i, int(8*p*(p-1)/2)) for i in fc_deconv]

enc_2 = [tf.reshape(i, [batch_size, 8, 1, int(p*(p-1)/2)]) for i in fc_deconv2] # height 8, width 1, channel p*(p-1)/2. NHWC

deconv1 = [0, 0]
deconv2 = [0, 0]
filter_deconv2 = [0, 0]
filter_deconv1 = [0, 0]

for i in xrange(2):
    filter_deconv1[i] = tf.Variable(tf.random_normal([w, 1, p, int(p*(p-1)/2)], stddev=0.5))
    deconv1[i] = tf.nn.conv2d_transpose(enc_2[i], filter_deconv1[i], output_shape=[batch_size, 32, 1, p], strides=[1, 4, 4, 1])
    filter_deconv2[i] = tf.Variable(tf.random_normal([w, 1, 1, p], stddev=0.5))
    deconv2[i] = tf.nn.conv2d_transpose(deconv1[i], filter_deconv2[i], output_shape=[batch_size, 64, 1, 1], strides=[1, 2, 2, 1])

decX = tf.reshape(deconv2[0], [batch_size, 64])
decY = tf.reshape(deconv2[1], [batch_size, 64])


y_true = tf.concat(X,Y)
y_pred = tf.concat(decX, decY)

x        = tf.slice(y_pred,[0,0],[-1,64])
y        = tf.slice(y_pred,[0,64], [-1,64])
IK       = np.fft.fftfreq(64)*1j
IK       = IK.astype(np.dtype('complex64'))
temp     = tf.complex(y_pred,0.0)
temp2    = tf.multiply(IK,tf.fft(tf.slice(temp,[0,0],[-1,64])))
dbydx    = tf.real(tf.ifft(temp2))
dbydy    = tf.real(tf.ifft(tf.multiply(IK,tf.fft(tf.slice(tf.complex(y_pred,0.0),[0, 64],[-1,64])))))

length   = tf.reduce_sum(tf.sqrt(tf.add(tf.square(dbydx),tf.square(dbydy))))
area     = tf.reduce_sum(tf.add(tf.multiply(x,dbydy),-1*tf.multiply(y,dbydx)))

r_x      = tf.slice(y_true,[0,0],[-1,64])
r_y      = tf.slice(y_true,[0,64], [-1,64])
r_dbydx  = tf.real(tf.ifft(tf.multiply(IK,tf.fft(tf.slice(tf.complex(y_true,0.0),[0,0],[-1, 64])))))
r_dbydy  = tf.real(tf.ifft(tf.multiply(IK,tf.fft(tf.slice(tf.complex(y_true,0.0),[0, 64],[-1,64])))))
r_length = tf.reduce_sum(tf.sqrt(tf.add(tf.square(r_dbydx),tf.square(r_dbydy))))
r_area   = tf.reduce_sum(tf.add(tf.multiply(r_x,r_dbydy),-1*tf.multiply(r_y,r_dbydx)))

c1       = tf.add_n([tf.reduce_mean(tf.pow(y_true - y_pred, 2)), alpha*regulariser])
c3       = parser.parse_args().length*(tf.pow(length-r_length,2))
c2       = parser.parse_args().roughness*tf.add_n([tf.reduce_mean(tf.square(dbydx)), tf.reduce_mean(tf.square(dbydy))])
c4       = parser.parse_args().area*(tf.pow((area-r_area)/r_area,2)) #1e-2

cost     = tf.add_n([c1 , c2, c3, c4])

optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)
#optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(cost)

# Initializing the variables
class saveHook(tf.train.SessionRunHook):
def after_run(self, run_context, fuckit):
    sess = run_context.session
    if global_step.eval(session=sess)%100 == 0:
        y_predict = sess.run(y_pred, feed_dict={X: train})
        y_predict_test = sess.run(y_pred, feed_dict={X: test})
        
        tosave = {  'training_y_act':train, 
                'training_y_pred':y_predict, 
                'type':"Leaky ReLu 128(input) -> "+str(n_hidden_1)+" -> "+str(n_hidden_2)+"-> "+str(n_hidden_3)+"-> "+str(n_hidden_4)+"-> "+str(n_hidden_5)+" (->"+ str(n_hidden_1) + "... Decode)", 
                'learning_rate': learning_rate, 
                'training_epochs':training_epochs, 
                'alpha_reg': alpha, 
                'test_y_act':test, 
                'test_y_pred':y_predict_test, 
                'costs':costs}
        sio.savemat("live" + str(parser.parse_args().res_n) + ".mat",tosave)
def end(self, sess):
    y_predict = sess.run(y_pred, feed_dict={X: train})
    y_predict_test = sess.run(y_pred, feed_dict={X: test})
    
    tosave = {  'training_y_act':train, 
            'training_y_pred':y_predict, 
            'type':"Leaky ReLu 128(input) -> "+str(n_hidden_1)+" -> "+str(n_hidden_2)+"-> "+str(n_hidden_3)+"-> "+str(n_hidden_4)+"-> "+str(n_hidden_5)+" (->"+ str(n_hidden_1) + "... Decode)", 
            'learning_rate': learning_rate, 
            'training_epochs':training_epochs, 
            'alpha_reg': alpha, 
            'test_y_act':test, 
            'test_y_pred':y_predict_test, 
            'costs':costs}
    sio.savemat(save_to+".mat",tosave)
    saver.save(sess, "./savedSession"+str(n_hidden_5)+"/model",global_step = global_step)

hooks=[tf.train.StopAtStepHook(last_step=training_epochs), saveHook()]

f.close()
f = open('costs' +"_"+hostname+"_"+ str(parser.parse_args().res_n), 'w')

costs = []
if job_name == 'ps':
server.join()
with tf.train.MonitoredTrainingSession(master=server.target,
                                       is_chief=(task_index == 0),
                                       checkpoint_dir="./timelySave"+str(n_hidden_5)+"/",
                                       hooks=hooks) as mon_sess:
print(server.target)
try:
    saver.restore(sess,"./savedSession"+str(n_hidden_5)+"/model")
except:
    pass
while not mon_sess.should_stop():
    _, c =  mon_sess.run([optimizer, cost], feed_dict={X: train})
    costs.append(c)
    #print(hostname +": "+ str(c))
    f.write(hostname +": "+ str(c)+'\n')

print("Optimization Finished!")
f.close()
