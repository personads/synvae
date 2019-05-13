import logging, os, sys

import numpy as np
import PIL
import tensorflow as tf

from models.base import BaseModel

class VisualVae(BaseModel):
    def __init__(self, img_height, img_width, img_depth, latent_dim, beta, batch_size, learning_rate):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.latent_dim = latent_dim
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = 0
        # set up computation graph
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_depth], name='images')


    def __repr__(self):
        res  = '<%s: ' % self.__class__.__name__
        res += str(self.images.shape[1:])
        res += ' -> %d (β=%s)' % (self.latent_dim, str(self.beta))
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res


    def build_encoder(self, images):
        latents, latent_dist = None, None
        return latents, latent_dist


    def build_decoder(self, latents):
        reconstructions = None
        return reconstructions


    def build(self):
        self.latents, means, sigmas = self.build_encoder(self.images)
        self.reconstructions = self.build_decoder(self.latents)
        # set up loss
        self.loss = self.calc_loss(self.images, self.reconstructions, means, sigmas)
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss)
        # debug info
        tf.summary.image('Originals', self.images, max_outputs=4)
        tf.summary.image('Reconstructions', self.reconstructions, max_outputs=4)
        tf.summary.scalar('Loss', self.loss)
        logging.info(self)


    def calc_loss(self, originals, reconstructions, means, sigmas):
        # MSE using flattened images for element-wise operations
        originals = tf.reshape(originals, [-1, self.img_width * self.img_height * self.img_depth])
        reconstructions = tf.reshape(reconstructions, [-1, self.img_width * self.img_height * self.img_depth])
        self.recon_loss = tf.reduce_mean(tf.reduce_sum(tf.square(originals - reconstructions), axis=-1))
        # KL divergence
        self.latent_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1. + sigmas - tf.square(means) - tf.exp(sigmas), axis=-1)) # mean KL over latent dims
        loss = self.recon_loss + self.beta * self.latent_loss
        return loss


    def save_image(self, image_array, path):
        image = PIL.Image.fromarray((image_array * 255).astype(np.uint8))
        image.save(path)


    def run_train_step(self, tf_session, batch):
        _, loss, recon_loss, latent_loss, summaries = tf_session.run([self.train_op, self.loss, self.recon_loss, self.latent_loss, self.merge_op], feed_dict={self.images: batch})
        losses = {'All': loss, 'MSE': recon_loss, 'KL': latent_loss}
        return losses, summaries


    def run_test_step(self, tf_session, batch, batch_idx, out_path, export_step=5):
        loss, recon_loss, latent_loss, reconstructions = tf_session.run([self.loss, self.recon_loss, self.latent_loss, self.reconstructions], feed_dict={self.images: batch})
        losses = {'All': loss, 'MSE': recon_loss, 'KL': latent_loss}
        # save original image and reconstruction
        if (export_step > 0) and ((batch_idx-1) % export_step == 0):
            if self.epoch == 1:
                self.save_image(batch[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_orig.png'))
            self.save_image(reconstructions[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_recon.png'))
        return losses


class BamVae(VisualVae):
    def __init__(self, latent_dim, beta, batch_size):
        super().__init__(img_height=256, img_width=256, img_depth=3, latent_dim=latent_dim, beta=beta, batch_size=batch_size, learning_rate=1e-3)


    def build_encoder(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv2)
        conv4 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv3)
        conv5 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv4)
        conv6 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv5)
        flat = tf.keras.layers.Flatten()(conv6)
        dense = tf.keras.layers.Dense(units=(self.img_width//64 * self.img_height//64 * 256), activation=tf.nn.relu)(flat)
        means = tf.keras.layers.Dense(self.latent_dim)(flat)
        sigmas = tf.keras.layers.Dense(self.latent_dim)(flat)
        epsilons = tf.random.normal((self.batch_size, self.latent_dim), mean=0., stddev=1.)
        latents = means + tf.exp(.5 * sigmas) * epsilons
        return latents, means, sigmas


    def build_decoder(self, latents):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))(latents)
        dense1 = tf.keras.layers.Dense(units=(self.img_width//64 * self.img_height//64 * 256), activation=tf.nn.relu)(inputs)
        shape = tf.keras.layers.Reshape(target_shape=(self.img_width//64, self.img_height//64, 256))(dense1)
        deconv1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(shape)
        deconv2 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(deconv1)
        deconv3 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(deconv2)
        deconv4 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(deconv3)
        deconv5 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(deconv4)
        deconv6 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(deconv5)
        recons = tf.keras.layers.Conv2DTranspose(filters=self.img_depth, kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)(deconv6)
        return recons


class CifarVae(VisualVae):
    def __init__(self, latent_dim, beta, batch_size):
        super().__init__(img_height=32, img_width=32, img_depth=3, latent_dim=latent_dim, beta=beta, batch_size=batch_size, learning_rate=1e-3)


    def build_encoder(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv2)
        flat = tf.keras.layers.Flatten()(conv3)
        dense = tf.keras.layers.Dense(units=(self.img_height//8 * self.img_width//8 * 256), activation=tf.nn.relu)(flat)
        means = tf.keras.layers.Dense(self.latent_dim)(flat)
        sigmas = tf.keras.layers.Dense(self.latent_dim)(flat)
        epsilons = tf.random.normal((self.batch_size, self.latent_dim), mean=0., stddev=1.)
        latents = means + tf.exp(.5 * sigmas) * epsilons
        return latents, means, sigmas


    def build_decoder(self, latents):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))(latents)
        dense1 = tf.keras.layers.Dense(units=(self.img_height//8 * self.img_width//8 * 256), activation=tf.nn.relu)(inputs)
        shape = tf.keras.layers.Reshape(target_shape=(self.img_height//8, self.img_width//8, 256))(dense1)
        deconv1 = tf.keras.layers.Conv2DTranspose(filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(shape)
        deconv2 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(deconv1)
        deconv3 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(deconv2)
        recons = tf.keras.layers.Conv2DTranspose(filters=self.img_depth, kernel_size=1, strides=1, padding='same', activation=tf.nn.sigmoid)(deconv3)
        return recons


class MnistVae(VisualVae):
    def __init__(self, latent_dim, beta, batch_size):
        super().__init__(img_height=28, img_width=28, img_depth=1, latent_dim=latent_dim, beta=beta, batch_size=batch_size, learning_rate=1e-3)


    def build_encoder(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(conv1)
        flat = tf.keras.layers.Flatten()(conv2)
        means = tf.keras.layers.Dense(self.latent_dim)(flat)
        sigmas = tf.keras.layers.Dense(self.latent_dim)(flat)
        epsilons = tf.random.normal((self.batch_size, self.latent_dim), mean=0., stddev=1.)
        latents = means + tf.exp(.5 * sigmas) * epsilons
        return latents, means, sigmas


    def build_decoder(self, latents):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))(latents)
        dense = tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu)(inputs)
        shape = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(dense)
        deconv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)(shape)
        deconv2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)(deconv1)
        recons = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.sigmoid)(deconv2)
        return recons
