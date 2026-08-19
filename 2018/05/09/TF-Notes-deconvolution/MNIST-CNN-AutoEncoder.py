#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 8 17:50:22 2018

@author: zhisheng
"""

import gzip
import os
GPU_IDX = "4,5"
os.environ["CUDA_VISIBLE_DEVICES"]=GPU_IDX
import tempfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pylab
import tensorflow as tf
from tensorflow.contrib.layers import flatten
# from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.examples.tutorials.mnist import input_data
# Shuffle arrays or sparse matrices in a consistent way
from sklearn.utils import shuffle
from sklearn.manifold import TSNE

embedding_dim = 2
outModelDir = './autoEncoder-cnn-with-l2norm-on-embedding-hdim{}/'.format(embedding_dim)


"""
Data Loading and Plotting
"""

dataPath='../dataset/MNIST_data/'
mnist = input_data.read_data_sets(dataPath, one_hot=True)
# read the images
images = mnist.train.images
images = images - 0.5 # Normalize
img_num, img_dim = images.shape
print('Images with shape {}'.format(images.shape))  # Images with shape (55000, 784)
# read the labels
labels1Hot = mnist.train.labels
print('labels1Hot.shape = {}'.format(labels1Hot.shape))
labels = np.argmax(labels1Hot,axis=1)
labels = labels[...,np.newaxis]
print('labels.shape = {}'.format(labels.shape))
n_classes = len(np.unique(labels))

"""
AutoEncoder Graph Construction
"""

# ===== First define the hyper-parameters and input output tensors
EPOCHS = 100
BATCH_SIZE = 32
rate = 0.0001
l2_weight = 0.001

img_height,img_width = 28, 28
cNum = 1
x = tf.placeholder(tf.float32, (None, img_height, img_width, cNum))
y = tf.placeholder(tf.float32, (None, img_height, img_width, cNum))

code = tf.placeholder(tf.float32, (None, embedding_dim))

# ===== Define the graph and construct it
# First, we define all the variables
ksize = 5
mu = 0
sigma = 0.1
layer_dim = {
    'conv1': 16,
    'conv2': 32,
    'embedded': embedding_dim
}
in_dim_for_embedded = 7*7*layer_dim['conv2']
with tf.variable_scope('AutoEncoder') as AEScope:
    initializer = tf.random_normal_initializer(mean=mu,stddev=sigma)
    conv1_w = tf.get_variable('conv1_w', shape=(ksize, ksize, cNum, layer_dim['conv1']), initializer=initializer)

    conv1_t_w = tf.get_variable('conv1_t_w', shape=(ksize, ksize, cNum, layer_dim['conv1']), initializer=initializer)

    conv2_w = tf.get_variable('conv2_w', shape=(ksize, ksize, layer_dim['conv1'], layer_dim['conv2']), initializer=initializer)

    conv2_t_w = tf.get_variable('conv2_t_w', shape=(ksize, ksize, layer_dim['conv1'], layer_dim['conv2']), initializer=initializer)

    embedded_w = tf.get_variable('embedded_w',shape=(in_dim_for_embedded,layer_dim['embedded']), initializer=initializer)

    reconstruct_w = tf.get_variable('reconstruct_w', shape=(ksize, ksize, cNum, cNum), initializer=initializer)
    
weights = {
    'conv1': conv1_w,
    'conv2': conv2_w,
    'conv1t': conv1_t_w,
    'conv2t': conv2_t_w,
    'embedded': embedded_w,
    'reconstruct': reconstruct_w
}

all_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=AEScope.name)
print('All collection ({}) having {} variables'.format(AEScope.name,len(all_collection)))

# Then we define the graph
def Encoder(x):
    print('Input x got shape=',x.shape)  # (None,28,28,1)
    # Layer 1 encode: Input = (batch_num, img_height, img_width, cNum). Output = (batch_num, img_height/2, img_width/2, layer_dim['conv1'])
    layer1_en = tf.nn.relu(tf.nn.conv2d(x, weights['conv1'], strides=[1, 1, 1, 1], padding='SAME'))
    # Avg Pooling
    layer1_en = tf.nn.avg_pool(layer1_en, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('After Layer 1, got shape=',layer1_en.shape)  # (None,14,14,32)

    # Layer 2 encode: Input = (batch_num, img_height/2, img_width/2, layer_dim['conv1']). Output = (batch_num, img_height/4, img_width/4, layer_dim['conv2'])
    layer2_en = tf.nn.relu(tf.nn.conv2d(layer1_en, weights['conv2'], strides=[1, 1, 1, 1], padding='SAME'))
    # Avg Pooling
    layer2_en = tf.nn.avg_pool(layer2_en, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID')
    print('After Layer 2, got shape=',layer2_en.shape)  # (None,7,7,64)

    # Layer embedded: Input = (batch_num, img_height/4 * img_width/4 * layer_dim['conv2']). Output = (batch_num, layer_dim['embedded'])
    flatten_in = flatten(layer2_en)
    embedded   = tf.matmul(flatten_in,weights['embedded'])
    print('embedded has shape=',embedded.shape)
    
    return embedded

def Decoder(embedded):
    # API: tf.nn.conv2d_transpose = (value, filter, output_shape, strides, padding='SAME', ...)
    bsize = tf.shape(embedded)[0]

    # Layer embedded decode: Input = (batch_num, layer_dim['embedded']). Output = (batch_num, in_dim_for_embedded)
    embedded_t = tf.matmul(embedded,weights['embedded'],transpose_b=True)
    embedded_t = tf.reshape(embedded_t,[-1, 7, 7, layer_dim['conv2']])
    print('embedded_t has shape=',embedded_t.shape)

    # Layer 2 decode: Input = (batch_num, 7, 7, layer_dim['conv2']). Output = (batch_num, 14, 14, layer_dim['conv1'])
    layer2_t = tf.nn.relu(tf.nn.conv2d_transpose(embedded_t,weights['conv2t'],[bsize, 14, 14, layer_dim['conv1']], [1, 2, 2, 1]))
    print('layer2_t has shape=',layer2_t.shape)

    # Layer 1 decode: Input = (batch_num, 14, 14, layer_dim['conv1']). Output = (batch_num, 28, 28, cNum)
    layer1_t = tf.nn.relu(tf.nn.conv2d_transpose(layer2_t,weights['conv1t'],[bsize, 28, 28, cNum], [1, 2, 2, 1]))
    print('layer1_t has shape=',layer1_t.shape)
    
    # Layer reconstruct: Input = batch_num x layer_dim['layer1']. Output = batch_num x img_dim.
    reconstruct = tf.nn.relu(tf.nn.conv2d(layer1_t, weights['reconstruct'], strides=[1, 1, 1, 1], padding='SAME')) - 0.5
    print('reconstruct has shape=',reconstruct.shape)
    
    return reconstruct

def AutoEncoder(x):    
    embedded = Encoder(x)
    reconstruct = Decoder(embedded)
    
    return [embedded, reconstruct]

[embedded_auto, reconstruct_auto] = AutoEncoder(x)
# for generative model used
reconstruct = Decoder(code)

all_collection = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
print('All collection ({}) having {} variables'.format(AEScope.name,len(all_collection)))


"""
Define loss and optimizer
"""

loss_op = tf.reduce_mean(tf.pow(tf.subtract(reconstruct_auto, y), 2.0)) + l2_weight* tf.reduce_mean(tf.pow(embedded_auto, 2.0))
optimizer = tf.train.AdamOptimizer(learning_rate = rate)
training_op = optimizer.minimize(loss_op)


"""
Run Session
"""

if not os.path.isdir(outModelDir):
    ### Train your model here.
    # mini-batch Adam training, will save model
    if not os.path.isdir(outModelDir):
        os.makedirs(outModelDir)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        num_examples = img_num
        
        print("Training...")
        print("")
        train_loss = np.zeros(EPOCHS)
        for i in range(EPOCHS):
            acc_train_loss = 0
            images_train, labels_train = shuffle(images, labels1Hot)
            for offset in range(0, num_examples, BATCH_SIZE):
                end = offset + BATCH_SIZE
                batch_img, batch_label = images_train[offset:end], labels_train[offset:end]
                # print('*'*10+' original batch_img.shape={}'.format(batch_img.shape))
                batch_img = np.reshape(batch_img,[-1, img_height, img_width, cNum])
                # print('*'*10+' reshaped batch_img.shape={}'.format(batch_img.shape))
                _, l, recon = sess.run([training_op, loss_op, reconstruct_auto], feed_dict={x: batch_img, y: batch_img})
                acc_train_loss += l/BATCH_SIZE
            train_loss[i] = acc_train_loss/len(range(0, num_examples, BATCH_SIZE))
            print("EPOCH {} ...".format(i+1))
            print("Train loss = {:.10f}".format(train_loss[i]))
            print("")
        saver.save(sess, outModelDir+'mnist-cnn-autoEncoder-model')
        print("Model saved")

    # Plot loss
    plt.figure(figsize=(7,5))
    plt.plot(range(EPOCHS), train_loss, 'b-^')
    plt.ylabel('Loss')
    plt.xlabel('Epochs')
    plt.axis('tight')
    plt.grid()
    plt.title('loss for epochs')
    plt.show()


"""
Plot Reconstruct images
"""
# Reduce number of images and labels
images = images[::10]
labels = labels[::10]

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint(outModelDir))
    embeddings, recon = sess.run([embedded_auto, reconstruct_auto], feed_dict={x: np.reshape(images,[-1,img_height, img_width, cNum])})

# Show some reconstructed images
plt.figure(figsize=(15,5))
plt.suptitle('Show some reconstruct images', fontsize=16)
recon += 0.5
for i in np.arange(2*7):
    random_idx = np.random.randint(0,len(recon))
    plt.subplot(2,7,i+1)
    plt.imshow(np.reshape(recon[random_idx],(28,28)),cmap='gray')
plt.show()


"""
Plot tSNE Embeddings
"""
print('output embeddings.shape = {}'.format(embeddings.shape))
# TSNE: http://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html
if (embedding_dim>2):
    embeddings_2d = TSNE(n_components=2).fit_transform(embeddings)
    print('output embeddings_2d.shape={}'.format(embeddings_2d.shape))
else:
    embeddings_2d = embeddings

# dot colors
cmap=list()
for i in np.arange(n_classes):
    cmap.append(np.random.rand(1,3))
print(cmap[0])
permute_idx = np.random.permutation(n_classes)
cmap = [cmap[i] for i in permute_idx]

pylab.figure(figsize=(10,10))
for i in np.arange(0,len(embeddings),5):
    x, y = (embeddings_2d[i,0],embeddings_2d[i,1])
    pylab.scatter(x,y,c=cmap[labels[i,0]])
pylab.show()


# """
# Do Image Generation by Decoder
# """

if (embedding_dim==2):
    codes = list()
    for xdim in np.arange(-1,1,0.2):
        for ydim in np.arange(-1,1,0.2):
            codes.append([xdim,ydim])
    print(len(codes))

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(outModelDir))
        sampled_img = sess.run(reconstruct,feed_dict={code:codes})

    sampled_img.shape
    plt.figure(figsize=(50,50))
    sampled_img += 0.5
    for i in np.arange(100):
        plt.subplot(10,10,i+1)
        plt.imshow(np.reshape(sampled_img[i],(28,28)),cmap='gray')
    plt.show()