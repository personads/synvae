import logging, os, sys

import numpy as np
import tensorflow as tf

from models.base import BaseModel

class Mine(BaseModel):
    def __init__(self, latent_dim, batch_size, shift=1, layer_size=128, learning_rate=1e-3):
        self.latent_dim = latent_dim
        self.layer_size = layer_size
        self.shift = shift
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = 0

        self.latents = tf.placeholder(tf.float32, [self.batch_size, self.latent_dim * 2], name='latents')


    def __repr__(self):
        res  = '<%s: ' % self.__class__.__name__
        res += '[%d, %d] -> ' % (self.latent_dim, self.latent_dim)
        res += '%d units -> ' % self.layer_size
        res += 'I(i;a)'
        res += '>'
        return res


    def build_mlp(self, inputs):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))(inputs)
        dense1 = tf.keras.layers.Dense(units=self.layer_size, activation=tf.nn.relu)(inputs)
        dense2 = tf.keras.layers.Dense(units=self.layer_size//2, activation=tf.nn.relu)(dense1)
        predictions = tf.keras.layers.Dense(1)(dense2)
        return predictions


    def build(self):
        vis_latents, aud_latents = self.latents[:, :self.latent_dim], self.latents[:, self.latent_dim:]
        shifted_latents = tf.concat([vis_latents, tf.concat([aud_latents[-self.shift:], aud_latents[:-self.shift]], axis=0)], axis=1)
        inputs = tf.concat([self.latents, shifted_latents], axis=0)

        self.predictions = self.build_mlp(inputs)
        # set up loss
        self.loss = self.calc_loss(self.predictions)
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss)
        # debug info
        tf.summary.scalar('Loss', self.loss)
        logging.info(self)


    def calc_loss(self, predictions):
        true_predictions = predictions[:self.latents.shape[0]]
        shifted_predictions = predictions[self.latents.shape[0]:]
        self.mi_true = true_predictions
        self.mi_shift = shifted_predictions
        loss = tf.reduce_mean(true_predictions) - tf.log(tf.reduce_mean(tf.exp(shifted_predictions)))
        loss = -loss
        return loss


    def run_train_step(self, tf_session, batch):
        _, cur_loss, mit, mis, summaries = tf_session.run([self.train_op, self.loss, self.mi_true, self.mi_shift, self.merge_op], feed_dict={self.latents: batch})
        return {'All': cur_loss,'MIT': np.mean(mit), 'MIS': np.mean(mis)}, summaries


    def run_test_step(self, tf_session, batch, batch_idx, out_path):
        cur_loss = tf_session.run(self.loss, feed_dict={self.latents: batch})
        return {'All': cur_loss}
