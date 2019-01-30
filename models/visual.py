import numpy as np
import PIL
import tensorflow as tf


class VisualVae:
    def __init__(self, img_height, img_width, img_depth, latent_dim, batch_size, learning_rate, verbose=True):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.verbose = verbose
        # set up computation graph
        self.images = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_depth])
        means, logvars = self.encode(self.images)
        latent = self.reparameterize(means, logvars)
        self.reconstructions = self.decode(latent)
        # set up loss calculation
        self.loss = self.calc_loss(self.images, self.reconstructions, means, logvars)
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss)
        # add summaries for TensorBoard
        tf.summary.image('Original', self.images, max_outputs=4)
        tf.summary.image('Reconstructions', self.reconstructions, max_outputs=4)
        tf.summary.scalar('Loss', self.loss)
        if self.verbose: print(self)


    def __repr__(self):
        res  = '<VisualVae: '
        res += str(self.images.shape[1:])
        res += ' -> ' + str(self.latent_dim)
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res


    def encode(self, images):
        mean, logvar = None, None
        return mean, logvar


    def reparameterize(self, mean, logvar):
        eps = tf.random_normal(shape=(self.batch_size, self.latent_dim), mean=0., stddev=1.) # batchsize, latent_dim (necessary while building graph)
        return eps * tf.exp(logvar) + mean


    def decode(self, latents):
        latents = None
        return latents


    def calc_loss(self, originals, reconstructions, means, logvars):
        originals_flat = tf.reshape(originals, (-1, self.img_height * self.img_width * self.img_depth))
        reconstructions_flat = tf.reshape(reconstructions, (-1, self.img_height * self.img_width * self.img_depth))

        recon_loss = tf.reduce_sum(tf.square(originals_flat - reconstructions_flat), axis=1)
        latent_loss = - 0.5 * tf.reduce_sum(1 + logvars - tf.square(means) - tf.exp(logvars), axis=1)
        loss = tf.reduce_mean(recon_loss + latent_loss)
        return loss


    def save(self, tf_session, path):
        save_path = tf.train.Saver().save(tf_session, path)
        if self.verbose: print("Saved model to '%s'." % save_path)


    def restore(self, tf_session, path):
        tf.train.Saver().restore(tf_session, path)
        if self.verbose: print("Restored model from '%s'." % path)


    def save_image(self, image_array, path):
        image = PIL.Image.fromarray((image_array * 255).astype(np.uint8))
        image.save(path)


class CifarVae(VisualVae):
    def __init__(self, latent_dim, batch_size, verbose=True):
        super().__init__(img_height=32, img_width=32, img_depth=3, latent_dim=latent_dim, batch_size=batch_size, learning_rate=1e-4, verbose=verbose)


    def encode(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=128, kernel_size=2, strides=2, padding='same', activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same', activation=tf.nn.relu)(conv2)
        flat = tf.keras.layers.Flatten()(conv3)
        dense = tf.keras.layers.Dense(units=(self.img_height//4 *  self.img_width//4 * 64), activation=tf.nn.relu)(flat)
        mean = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.relu)(dense)
        logvar = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.relu)(dense)
        return mean, logvar


    def decode(self, latents):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))(latents)
        dense1 = tf.keras.layers.Dense(units=(self.img_height//4 *  self.img_width//4 * 32), activation=tf.nn.relu)(inputs)
        dense2 = tf.keras.layers.Dense(units=(self.img_height//4 *  self.img_width//4 * 64), activation=tf.nn.relu)(dense1)
        shape = tf.keras.layers.Reshape(target_shape=(self.img_height//4, self.img_width//4, 64))(dense2)
        deconv1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(shape)
        deconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(deconv1)
        deconv3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(deconv2)
        recons = tf.keras.layers.Conv2DTranspose(filters=self.img_depth, kernel_size=2, strides=1, padding="same", activation=tf.nn.sigmoid)(deconv3)
        return recons


class MnistVae(VisualVae):
    def __init__(self, latent_dim, batch_size, verbose=True):
        super().__init__(img_height=28, img_width=28, img_depth=1, latent_dim=latent_dim, batch_size=batch_size, learning_rate=1e-4, verbose=verbose)


    def encode(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(conv1)
        flat = tf.keras.layers.Flatten()(conv2)
        mean = tf.keras.layers.Dense(self.latent_dim)(flat)
        logvar = tf.keras.layers.Dense(self.latent_dim)(flat)
        return mean, logvar


    def decode(self, latents):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))(latents)
        dense = tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu)(inputs)
        shape = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(dense)
        deconv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)(shape)
        deconv2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)(deconv1)
        recons = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.sigmoid)(deconv2)
        return recons
