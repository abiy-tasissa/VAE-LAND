# ------------------------------------------------------------------------------------------
# This code implements a VAE architecture to learn latent features of hyperspectral images
# The output of this code is csv files which can be used as an input to the associated
# MATLAB script
# ------------------------------------------------------------------------------------------
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
#from custom_data import get_data
from sklearn import svm
from sklearn.cluster import KMeans
import tensorflow.keras.backend as backend
from scipy.io import loadmat
from sklearn.model_selection import train_test_split



class VAE(tf.keras.Model):
  def __init__(self, latent_dim):
    super(VAE, self).__init__()
    self.latent_dim = latent_dim
    self.inference_net = tf.keras.Sequential(
      [
          tf.keras.layers.InputLayer(input_shape=INPUT_SHAPE),
          tf.keras.layers.Dense(
              units = 128, activation='relu'),
          tf.keras.layers.Dense(
              units = 128, activation='relu'),
          tf.keras.layers.Dense(
              units = 128, activation='relu'),
	  # No activation
          tf.keras.layers.Dense(latent_dim + latent_dim),
      ]
    )

    self.generative_net = tf.keras.Sequential(
        [
          tf.keras.layers.InputLayer(input_shape=(latent_dim,)),
          tf.keras.layers.Dense(units = 128, activation='relu'),
          tf.keras.layers.Dense(units = 128, activation='relu'),
	  tf.keras.layers.Dense(units = 128, activation='relu'),
          tf.keras.layers.Dense(units = INPUT_SHAPE),

        ]
    )

  @tf.function
  def sample(self, eps=None):
    if eps is None:
      eps = tf.random.normal(shape=(100, self.latent_dim))
    return self.decode(eps, apply_sigmoid=True)

  def encode(self, x):
    mean, logvar = tf.split(self.inference_net(x), num_or_size_splits=2, axis=1)
    return mean, logvar

  def reparameterize(self, mean, logvar):
    eps = tf.random.normal(shape=mean.shape)
    return eps * tf.exp(logvar * .5) + mean

  def decode(self, z, apply_sigmoid=False):
    logits = self.generative_net(z)
    if apply_sigmoid:
      probs = tf.sigmoid(logits)
      return probs

    return logits

optimizer = tf.keras.optimizers.Adam(1e-4)

def log_normal_pdf(sample, mean, logvar, raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
      -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
      axis=raxis)

@tf.function
def compute_loss(model, x):
  mean, logvar = model.encode(x)
  z = model.reparameterize(mean, logvar)
  x_logit = model.decode(z)
 
  cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=tf.cast(x,tf.float32))
  logpx_z = -tf.reduce_sum(cross_ent, axis=[1])
  logpz = log_normal_pdf(z, 0., 0.)
  logqz_x = log_normal_pdf(z, mean, logvar)
  return -tf.reduce_mean(logpx_z + logpz - logqz_x)

#@tf.function
def compute_apply_gradients(model, x, optimizer):
  with tf.GradientTape() as tape:
    loss = compute_loss(model, x)
  gradients = tape.gradient(loss, model.trainable_variables)
  optimizer.apply_gradients(zip(gradients, model.trainable_variables))
  
  

epochs = 50


# Load a data of interest and labels
X = loadmat('SalinasA.mat')['salinasA']
y = loadmat('SalinasA_gt.mat')['salinasA_gt']
X = np.reshape(X,(X.shape[0]*X.shape[1],X.shape[2]))
y = np.reshape(y,(y.shape[0]*y.shape[1],-1))

# Map labels suitable for input to the LAND code
y[y==10] = 2
y[y==11] = 3
y[y==12] = 4
y[y==13] = 5
y[y==14] = 6

from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
normalizer = preprocessing.Normalizer()

X = min_max_scaler.fit_transform(X)
#X = normalizer.fit_transform(X)

# No need to split the training and test data    
#X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=0,stratify=y)
X_train = X;
X_test = X_train;
y_train = y;
y_test = y_train;


INPUT_SHAPE = X_train.shape[1]

# Latent dimension - this is an important parameter
latent_dims = 40

TRAIN_BUF = X_train.shape[0]
BATCH_SIZE = 100
TEST_BUF = X_test.shape[0]

train_dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.from_tensor_slices(X_test).shuffle(TEST_BUF).batch(BATCH_SIZE)
    

    

backend.clear_session()
model = VAE(latent_dims)
for epoch in range(1, epochs + 1):
    print(epoch)
    for train_x in train_dataset:
        compute_apply_gradients(model, train_x, optimizer)
        
z_train,_ = np.array(model.encode(X_train))
z_test,_ = np.array(model.encode(X_test))

# Write outputs to a text file
np.savetxt("salinas_train_dim"+str(latent_dims)+".csv",z_train,delimiter=',')
np.savetxt("salinas_train_label_dim"+str(latent_dims)+".csv",y_train,delimiter=',')
np.savetxt("salinas_test_dim"+str(latent_dims)+".csv",z_test,delimiter=',')
np.savetxt("salinas_test_label_dim"+str(latent_dims)+".csv",y_test,delimiter=',')
 
    
