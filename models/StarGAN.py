# Based on the StarGAN paper and their PyTorch implementation
# https://github.com/yunjey/stargan
# @InProceedings{StarGAN2018,
# author = {Choi, Yunjey and Choi, Minje and Kim, Munyoung and Ha, Jung-Woo and Kim, Sunghun and Choo, Jaegul},
# title = {StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation},
# booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
# month = {June},
# year = {2018}
# }

import tensorflow as tf
from tensorflow import layers, backend
from tensorflow import Model, Sequential, load_model
from tensorflow import Adam, InstanceNormalization
from tensorflow import LeakyReLU, ReLU, Conv2D, UpSampling2D, Input, Reshape, Concatenate, ZeroPadding2D, Lambda
import numpy as np
import cv2
import glob
import re
import random
from functools import partial

# Model architecture as described in the StarGAN paper.
class StarGAN():
  def __init__(self, config):
    self.input_shape = config.input_shape
    self.num_epochs = config.num_epochs
    self.num_c = config.num_c
    self.d_lr = config.d_lr
    self.g_lr = config.g_lr
    self.beta1 = config.beta1
    self.beta2 = config.beta2
    self.lambda_cls = config.lambda_cls
    self.lambda_rec = config.lambda_rec
    self.lambda_gp = config.lambda_gp
    self.batch_size = config.batch_size
    self.decay_lr = config.decay_lr
  
  def build_generator(self):  
    def conv2d(x, filters, kernel_size, strides, padding):
      x = ZeroPadding2D(padding=padding)(x)
      x = Conv2D(filters, kernel_size, strides, padding='valid', use_bias=False)(x)
      x = ReLU()(x)
      x = InstanceNormalization(axis=-1)(x)
      return x
    
    def deconv2d(x, filters, kernel_size, strides, padding):
      x = UpSampling2D(2)(x)
      x = Conv2D(filters, kernel_size, strides, padding='same', use_bias=False)(x)
      x = ReLU()(x)
      x = InstanceNormalization(axis=-1)(x)
      return x

    def down_sampling(x):
      d1 = conv2d(x, 64, 7, 1, 3)
      d2 = conv2d(d1, 128, 4, 2, 1)
      d3 = conv2d(d2, 256, 4, 2, 1)
      return d3

    def bottleneck(x):
      for _ in range(6):
        x = conv2d(x, 256, 3, 1, 1)
      return x
    
    def up_sampling(x):
      u1 = deconv2d(x, 128, 4, 1, 1)
      u2 = deconv2d(u1, 64, 4, 1, 1)
      return u2

    def output_conv(x):
      x = ZeroPadding2D(padding=3)(x)
      x = Conv2D(filters=3, kernel_size=7, strides=1, padding='valid', activation='tanh', use_bias=False)(x)
      return x
    
    input_img = Input(self.input_shape)
    input_c = Input((self.num_c,))
    c = Lambda(lambda x: backend.repeat(x, 128**2))(input_c)
    c = Reshape(self.input_shape)(c)
    x = Concatenate()([input_img, c])
    down_sampled = down_sampling(input_img)
    bottlenecked = bottleneck(down_sampled)
    up_sampled = up_sampling(bottlenecked)
    out = output_conv(up_sampled)
    return Model(inputs=[input_img, input_c], outputs=out)

  def build_discriminator(self):
    def conv2d(x, filters, kernel_size, strides, padding):
      x = ZeroPadding2D(padding=padding)(x)
      x = Conv2D(filters, kernel_size, strides, padding='valid', use_bias=False)(x)
      x = LeakyReLU(0.01)(x)
      return x
    
    input_img = Input(self.input_shape)
    x = input_img
    filters = 64
    for _ in range(6):
      x = conv2d(x, filters, 4, 2, 1)
      filters = filters*2

    out_cls = Conv2D(self.num_c, 2, 1, padding='valid', use_bias=False)(x)
    out_cls = Reshape((self.num_c,))(out_cls)
    x = ZeroPadding2D(padding=1)(x)
    out_src = Conv2D(1, 3, 1, padding='valid', use_bias=False)(x)
    return Model(inputs=input_img, outputs=[out_src, out_cls])
  
  def classification_loss(self, y_expected, y_pred):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_expected, logits=y_pred))

  def reconstruction_loss(self, y_expected, y_pred):
    return backend.mean(backend.abs(y_expected - y_pred))

  def wasserstein_loss(self, y_expected, y_pred):
    return backend.mean(y_expected*y_pred)
  
  # Implemented as recommended by Keras.
  # https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
  def gradient_penalty_loss(self, y_expected, y_pred, averaged_samples):
    gradients_squared = backend.square(backend.gradients(y_pred, averaged_samples)[0])
    shape = len(gradients_squared.shape)
    gradients_squared_sum = backend.sum(gradients_squared, axis=np.arange(1, shape))
    gradient_penalty = backend.square(1 - backend.sqrt(gradients_squared_sum))
    return backend.mean(gradient_penalty)


  # Returns random weighted average of two tensors.
  def random_weighted_avg(self, real, fake, batch_size):
    weights = backend.random_uniform((batch_size, 1, 1, 1))
    return (weights * real) + ((1 - weights) * fake)

  # Used to smooth the label in classification loss while training D
  def smooth_y(self, y):
    return y*backend.random_normal((self.batch_size, y.shape[1]), 1, 0.001)
    # return y

  def build_model(self):
    self.G = self.build_generator()
    self.D = self.build_discriminator()
    
    # Make D trainable and G not trainable for training of discriminator
    for layer in self.G.layers:
      layer.trainable = False
    self.G.trainable = False
    for layer in self.D.layers:
      layer.trainable = True
    self.D.trainable = True
    
    # Inputing real image into discriminator
    x_real_D = Input(self.input_shape)
    out_src_real_D, out_cls_real_D = self.D(x_real_D)
    
    ## Using G to create a fake image and putting that into the discriminator
    # Target class to transfer G's input to
    label_trg_D = Input((self.num_c,))

    # Fake images created by G
    x_fake_D = self.G([x_real_D, label_trg_D])

    # D's output from the fake images
    out_src_fake_D, out_cls_fake_D = self.D(x_fake_D)
    
    # Random weighted average of the real and fake images
    x_hat = self.random_weighted_avg(x_real_D, x_fake_D, self.batch_size)

    # D's output from the averaged input for use in calculating gradient penalty loss
    out_src_x_hat, _ = self.D(x_hat)

    # partial_gp_loss requires the averaged samples as weights but Keras will only supply 
    # y_pred and y_true for a loss function.
    # See https://github.com/keras-team/keras-contrib/blob/master/examples/improved_wgan.py
    partial_gp_loss = partial(self.gradient_penalty_loss, averaged_samples=x_hat)
    partial_gp_loss.__name__ = 'gradient_penalty'

    # Smooth label
    out_cls_real_smoothed_D = self.smooth_y(out_cls_real_D)

    # Create train_D for training the discriminator with frozen generator weights
    self.train_D = Model([x_real_D, label_trg_D], [out_src_real_D, out_cls_real_smoothed_D, out_src_fake_D, out_src_x_hat])
    self.train_D.compile(loss=[self.wasserstein_loss, self.classification_loss, self.wasserstein_loss, partial_gp_loss],
                         optimizer=Adam(lr=self.d_lr, beta_1=self.beta1, beta_2=self.beta2),
                         loss_weights=[1, self.lambda_cls, 1, self.lambda_gp])
    
    # Now set up to train G
    for layer in self.G.layers:
      layer.trainable = True
    for layer in self.D.layers:
      layer.trainable = False
    self.G.trainable = True
    self.D.trainable = False

    # Define inputs
    x_real_G = Input(self.input_shape)
    original_label_G = Input((self.num_c,))
    target_label_G = Input((self.num_c,))

    # Create fake image with G
    x_fake_G = self.G([x_real_G, target_label_G])

    # Get outputs of D from fake image
    out_src_fake_G, out_cls_fake_G = self.D(x_fake_G)

    # Pass the image through G with its original labels
    # for calculating reconstruction loss
    x_rec_G = self.G([x_fake_G, original_label_G])

    # Create train_G for training generator with frozen discriminator weights
    self.train_G = Model([x_real_G, original_label_G, target_label_G], [out_src_fake_G, out_cls_fake_G, x_rec_G])
    self.train_G.compile(loss=[self.wasserstein_loss, self.classification_loss, self.reconstruction_loss],
                         optimizer=Adam(lr=self.g_lr, beta_1=self.beta1, beta_2=self.beta2),
                         loss_weights=[1, self.lambda_cls, self.lambda_rec])
    
    # Load batches of training data
    self.training_data = DataLoader()
    self.training_data.load_data()

    for layer in self.D.layers:
      layer.trainable = True
    self.D.trainable = True

  def smooth_label(self, y):
    return y*np.random.normal(0.95,0.1,y.shape)

  def train(self):
    loader = self.training_data.get_loader(self.batch_size, self.training_data.train_data)
    
    # Discriminator labels for fake images, real images, and dummy labels for gradient penalty
    fake_y = np.ones((self.batch_size, 2, 2, 1), dtype=np.float32)
    real_y = -fake_y
    dummy_y = np.zeros((self.batch_size, 2, 2, 1), dtype=np.float32)
    num_iter = len(loader)
    for epoch in range(self.num_epochs):
      # Decay learning rate each epoch
      if self.decay_lr:
        backend.set_value(self.train_D.optimizer.lr, self.d_lr*((self.num_epochs - epoch)/(self.num_epochs)))
        backend.set_value(self.train_G.optimizer.lr, self.g_lr*((self.num_epochs - epoch)/(self.num_epochs)))
      print(len(loader))
      D_losses = np.zeros((4))
      G_losses = np.zeros((3))
      random.shuffle(loader)
      for i, batch in enumerate(loader):

        imgs, original_labels = batch
        target_labels = self.training_data.random_labels(self.batch_size)

        # Add noise to input images
        imgs_D = imgs + np.random.normal(0,0.005,imgs.shape)
        
        # Train Discriminator on every iteration.
        D_loss = self.train_D.train_on_batch(x = [imgs_D, target_labels], y = [self.smooth_label(real_y), original_labels, self.smooth_label(fake_y), dummy_y])
        D_losses += np.array([D_loss[1], D_loss[3], D_loss[2], D_loss[4]])

        # Train Generator on every fifth iteration.
        if (i + 1) % 5 == 0:
          G_loss = self.train_G.train_on_batch(x = [imgs, original_labels, target_labels], y = [self.smooth_label(real_y), target_labels, imgs])
          G_losses += np.array([G_loss[1], G_loss[3], G_loss[2]])
          
      print(f"Epoch: {epoch}")
      print(f"\tMean Epoch Loss: D/loss_real = [{D_losses[0]/num_iter:.4f}], D/loss_fake = [{D_losses[1]/num_iter:.4f}], D/loss_cls =  [{D_losses[2]/num_iter:.4f}], D/loss_gp = [{D_losses[3]/num_iter:.4f}]")
      # if (epoch + 1) % 5 == 0:
      print(f"\tMean Epoch Loss: G/loss_fake = [{G_losses[0]/num_iter:.4f}], G/loss_rec = [{G_loss[1]/num_iter:.4f}], G/loss_cls = [{G_loss[2]/num_iter:.4f}]") 
      
      # Save weights every five epochs
      if (epoch + 1) % 5 == 0:
        self.G.save_weights('drive/My Drive/G_weights.hdf5')
        self.D.save_weights('drive/My Drive/D_weights.hdf5')
        self.train_D.save_weights('drive/My Drive/train_D_weights.hdf5')
        self.train_G.save_weights('drive/My Drive/train_G_weights.hdf5')

class Config(object):
    num_c = 7
    input_shape = (128,128,3)
    batch_size = 16
    num_epochs = 100
    d_lr = 0.0001
    g_lr = 0.0001
    beta1 = 0.5
    beta2 = 0.999
    decay_lr = False

    # Weights for the classification, reconstruction, and gradient penalty losses
    lambda_cls = 1
    lambda_rec = 10
    lambda_gp = 10

class DataLoader():

    def __init__(self, name='', path=''):
        self.dataset_name = name    
        self.dataset_path = path   
        
        # emotions as described in filenames, used for creating labels
        self.emotions = ['ne', 'af', 'an', 'di', 'ha', 'sa', 'su']
        # positions as described in filenames
        self.positions = ['s.', 'fl', 'hl', 'hr', 'fr']

        # This is randomly permuted to create a target label
        self.emotion_label = np.array([0,1,0,0,0,0,0])
        self.position_label = np.array([0,1,0])
        self.train_data = []
        self.custom_data = []
                
    
    # Used to sort inputs alphabetically
    def digits_in_string(self, text):
        convert = lambda text: int(text) if text.isdigit() else text.lower()
        return [convert(c) for c in re.split(r'(\d+)', text)]

    # Create random target labels for a batch
    def random_labels(self, batch_size):
        random_labels = np.zeros((batch_size, 7))
        for i in range(batch_size):
            random_labels[i] = np.random.permutation(self.emotion_label)
        return random_labels
    
    # Create a label for the image based on the emotion specified in the file name
    def create_label(self, filename):
        labels = np.zeros(7)
        img_name = filename.split('/')[-1]
        
        img_emotion = ''.join(img_name[4:6]).lower()
        labels[0 + self.emotions.index(img_emotion)] = 1

        return labels

    # Find 'num' faces with a neutral expression facing forward.
    # Used for testing.
    def load_test_data(self, num=10):
        img_filenames = glob.glob(self.dataset_path)
        num_found = 0
        i = 0
        while num_found < 10:
            img_name = img_filenames[i].split('/')[-1]
            emotion = ''.join(img_name[4:6]).lower()
            position = ''.join(img_name[6]).lower()
            if emotion == 'ne' and position == 's':
                img = cv2.imread(img_filenames[i])/127.5 - 1
                try:
                    original_labels = self.create_label(img_filenames[i])
                    flip_prob = np.random.rand()
                    if flip_prob > 0.5:
                        img = cv2.flip(img, 1)
                    self.custom_data.append([img, original_labels])
                    num_found += 1
                except:
                    i += 1
                    continue
            i += 1

    def load_data(self, num=4900):
        img_filenames = glob.glob(self.dataset_path)

        for i in range(num):
            img_name = img_filenames[i].split('/')[-1]
            position = ''.join(img_name[6:8]).lower()
     
            # Exclude images with position facing completely left or right
            if position != 'fl' and position != 'fr':

                # Normalize inputs between -1 and 1
                img = cv2.imread(img_filenames[i])/127.5 - 1
                try:
                    original_labels = self.create_label(img_filenames[i])
                    flip_prob = np.random.rand()
                    # Flip half of images for data augmentation
                    if flip_prob > 0.5:
                        img = cv2.flip(img, 1)
                    self.train_data.append([img, original_labels])
                except:
                    continue
     
    def get_loader(self, batch_size, data=None):
        if data == None:
            data = self.train_data
        total_batches = int(len(data)//batch_size)
        img_shape, label_shape = (batch_size, 128, 128, 3), (batch_size, 7)
        batches = []

        # Create batches of images and original_labels of size batch_size
        for i in range(total_batches):
            batch = data[i*batch_size:i*batch_size + batch_size]
            imgs = []
            original_labels = []
            target_labels = []
            for b in batch:
                imgs.append(b[0])
                original_labels.append(b[1])
            imgs = np.array(imgs)
            original_labels = np.array(original_labels)
            if imgs.shape == img_shape and original_labels.shape == label_shape:
                batches.append([imgs, original_labels])
        print('DATA LOADED\n')
        return batches
     