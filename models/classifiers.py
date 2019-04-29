import logging, os, sys

import tensorflow as tf

from models.base import BaseModel

class VisualCnn(BaseModel):
    def __init__(self, img_height, img_width, img_depth, num_labels, batch_size, learning_rate):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.num_labels = num_labels
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = 0
        # set up computation graph
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_depth], name='images')
        self.labels = tf.placeholder(tf.int32, [self.batch_size], name='labels')


    def __repr__(self):
        res  = '<VisualCnn: '
        res += str(self.images.shape[1:])
        res += ' -> (%d classes)' % self.num_labels
        res += '>'
        return res


    def build_cnn(self, images):
        predictions = None
        return predictions


    def build(self):
        self.predictions = self.build_cnn(self.images)
        # set up loss
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.predictions, labels=self.labels))
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss)
        # debug info
        tf.summary.image('Images', self.images, max_outputs=4)
        tf.summary.tensor_summary('Truth', self.labels)
        tf.summary.tensor_summary('Predictions', self.predictions)
        tf.summary.scalar('Loss', self.loss)
        logging.info(self)


    def run_train_step(self, tf_session, batch):
        _, cur_loss, summaries = tf_session.run([self.train_op, self.loss, self.merge_op], feed_dict={self.images: batch['images'], self.labels: batch['labels']})
        return {'All': cur_loss}, summaries


    def run_test_step(self, tf_session, batch, batch_idx, out_path):
        cur_loss = tf_session.run(self.loss, feed_dict={self.images: batch['images'], self.labels: batch['labels']})
        return {'All': cur_loss}


class CifarCnn(VisualCnn):
    def __init__(self, batch_size):
        super().__init__(img_height=32, img_width=32, img_depth=3, num_labels=10, batch_size=batch_size, learning_rate=1e-4)


    def build_cnn(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv2)
        flat = tf.keras.layers.Flatten()(conv3)
        dense = tf.keras.layers.Dense(units=(self.img_height//8 * self.img_width//8 * 256), activation=tf.nn.relu)(flat)
        predictions = tf.keras.layers.Dense(self.num_labels, activation=tf.nn.softmax)(dense)
        return predictions


class MnistCnn(VisualCnn):
    def __init__(self, batch_size):
        super().__init__(img_height=28, img_width=28, img_depth=1, num_labels=10, batch_size=batch_size, learning_rate=1e-4)


    def build_cnn(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(conv1)
        flat = tf.keras.layers.Flatten()(conv2)
        predictions = tf.keras.layers.Dense(self.num_labels, activation=tf.nn.softmax)(flat)
        return predictions
