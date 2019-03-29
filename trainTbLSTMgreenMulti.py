# -*- coding: utf-8 -*-

from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from makeDataset import datasetMaker
import math

# Training Parameters
learning_rate = 0.001
training_steps = 500000
batch_size = 20
display_step = 200
lossMatrix = np.zeros((training_steps))
lossMatrixMerge = np.zeros((3148))
lossOrgList = []

# Network Parameters
num_input = 49
timesteps = 10
num_hidden = 128 # hidden layer num of features
num_hidden_layer = 3
num_output = 49

datasetMaker = datasetMaker()

dataset, datasetAb = datasetMaker.datasetGreen()

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_output])
keep_prob = tf.placeholder("float")

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_output]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_output]))
}

# RMSの計算
def calRms(data):
    square = np.power(data,2)
    rms = math.sqrt(sum(square)/len(data))
    return rms

def make_batch(batch_size):
    batch = np.zeros((batch_size, 10, 49))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 49))
    output = np.array(output, dtype=np.float32)
    for i in range(batch_size):
        index = random.randint(0, 950)
        for k in range(49):
            output[i, k] = dataset[index + 10][k + 1]
        for k in range(49):
            for j in range(10):
                batch[i, j, k] = dataset[index + j][k + 1]
    for i in range(49):
        for k in range(batch_size):
            rms = calRms(batch[k, :, i])
            batch[k, :, i] /= rms
            output[k, i] /= rms
    return batch, output

def make_batch_normal(batch_size, startF):
    batch = np.zeros((batch_size, 10, 49))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 49))
    output = np.array(output, dtype=np.float32)
    for i in range(batch_size):
        index = startF
        for k in range(49):
            output[i, k] = dataset[index + 10][k + 1]
        for k in range(49):
            for j in range(10):
                batch[i, j, k] = dataset[index + j][k + 1]
    for i in range(49):
        for k in range(batch_size):
            rms = calRms(batch[k, :, i])
            batch[k, :, i] /= rms
            output[k, i] /= rms
    return batch, output

def make_batch_abnormal(batch_size, startF):
    batch = np.zeros((batch_size, 10, 49))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 49))
    output = np.array(output, dtype=np.float32)
    for i in range(batch_size):
        index = startF
        for k in range(49):
            output[i, k] = datasetAb[index + 10][k + 1]
        for k in range(49):
            for j in range(10):
                batch[i, j, k] = datasetAb[index + j][k + 1]
    for i in range(49):
        for k in range(batch_size):
            rms = calRms(batch[k, :, i])
            batch[k, :, i] /= rms
            output[k, i] /= rms
    return batch, output


x = tf.unstack(X, timesteps, 1)

def get_a_cell(lstm_size, keep_prob):
    lstm = rnn.BasicLSTMCell(lstm_size, forget_bias=1.0)
    drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
    return drop
    
    
with tf.name_scope('lstm'):
     cell = tf.nn.rnn_cell.MultiRNNCell(
     [get_a_cell(num_hidden, keep_prob) for _ in range(num_hidden_layer)]
     )

outputs, states = rnn.static_rnn(cell, x, dtype=tf.float32)

logits = tf.nn.sigmoid(tf.matmul(outputs[-1], weights['out']) + biases['out'])
prediction = logits

# Define loss and optimizer
lossOrg = logits - Y
loss_op = tf.reduce_mean(tf.square(lossOrg))

optimizer = tf.train.AdamOptimizer(1e-4)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = make_batch(batch_size)
        
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y, keep_prob: 0.5})
        # Calculate batch loss and accuracy
        loss = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})
        lossMatrix[step-1] = loss[0]
        if step % display_step == 0 or step == 1:
            
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss[0]))
         
            
    for step in range(998):
        batch_x, batch_y = make_batch_normal(batch_size, step)
        lossOrgList.append(sess.run([lossOrg], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})[0])
        lossMatrixMerge[step] = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})[0]
        
    for step in range(2150):
        batch_x, batch_y = make_batch_abnormal(batch_size, step)
        lossOrgList.append(sess.run([lossOrg], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})[0])
        lossMatrixMerge[step+998] = sess.run([loss_op], feed_dict={X: batch_x, Y: batch_y, keep_prob: 1.0})[0]

    print("Optimization Finished!")

LSTMdict = {"lossMatrix":lossMatrix, "lossMatrixMerge":lossMatrixMerge, "lossOrgList":lossOrgList}
pickle.dump(LSTMdict, open('LSTMdict.pickle', mode='wb'))
