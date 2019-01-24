import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


class VisualVae:
    def __init__(self, img_height, img_width, img_depth, latent_dim, batch_size, learning_rate, verbose=True):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.verbose = verbose
        # set up encoder and decoder networks
        self.encoder = self.make_encoder()
        self.decoder = self.make_decoder()
        # set up computation graph
        self.images = tf.placeholder(tf.float32, [None, self.img_height, self.img_width, self.img_depth])
        mean, logvar = self.encode(self.images)
        latent = self.reparameterize(mean, logvar)
        self.reconstructions = self.decode(latent)
        # set up loss calculation
        self.loss = self.calc_loss(self.images, self.reconstructions, latent, mean, logvar)
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss)
        # add summaries for TensorBoard
        tf.summary.image('Original', self.images, max_outputs=4)
        tf.summary.image('Reconstructions', self.reconstructions, max_outputs=4)
        tf.summary.scalar('Loss', self.loss)


    def make_encoder(self):
        return None


    def make_decoder(self):
        return None


    def encode(self, images):
        mean, logvar = tf.split(self.encoder(images), num_or_size_splits=2, axis=1)
        return mean, logvar


    def reparameterize(self, mean, logvar):
        eps = tf.random_normal(shape=(self.batch_size, self.latent_dim)) # batchsize, latent_dim (necessary while building graph)
        return eps * tf.exp(logvar * .5) + mean


    def decode(self, z, apply_sigmoid=False):
        logits = self.decoder(z)
        if apply_sigmoid:
            probs = tf.sigmoid(logits)
            return probs
        return logits


    def log_normal(self, sample, mean, logvar, raxis=1):
        log2pi = tf.log(2. * np.pi)
        return tf.reduce_sum(
                -.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi),
                axis=raxis)


    def calc_loss(self, original, reconstruction, latent, mean, logvar):
        cross_ent = tf.nn.sigmoid_cross_entropy_with_logits(logits=reconstruction, labels=original)
        logpx_z = -tf.reduce_sum(cross_ent, axis=[1, 2, 3])
        logpz = self.log_normal(latent, 0., 0.)
        logqz_x = self.log_normal(latent, mean, logvar)
        return -tf.reduce_mean(logpx_z + logpz - logqz_x)


    def save(self, tf_session, path):
        save_path = tf.train.Saver().save(tf_session, path)
        if self.verbose: print("Saved model to '%s'." % save_path)


    def restore(self, tf_session, path):
        tf.train.Saver().restore(tf_session, path)
        if self.verbose: print("Restored model from '%s'." % path)


    def save_image(self, image, path):
        color_map = None
        if len(image.shape) < 3:
            color_map = 'gray'
        plt.imshow(image, cmap=color_map)
        plt.axis('off')
        plt.savefig(path)


class MnistVae(VisualVae):
    def __init__(self, latent_dim, batch_size, verbose=True):
        super().__init__(img_height=28, img_width=28, img_depth=1, latent_dim=latent_dim, batch_size=batch_size, learning_rate=1e-4, verbose=verbose)


    def make_encoder(self):
        return tf.keras.Sequential(
            [
                    tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth)),
                    tf.keras.layers.Conv2D(
                            filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),
                    tf.keras.layers.Conv2D(
                            filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu),
                    tf.keras.layers.Flatten(),
                    # No activation
                    tf.keras.layers.Dense(self.latent_dim + self.latent_dim),
            ]
        )


    def make_decoder(self):
        return tf.keras.Sequential(
            [
                tf.keras.layers.InputLayer(input_shape=(self.latent_dim,)),
                tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu),
                tf.keras.layers.Reshape(target_shape=(7, 7, 32)),
                tf.keras.layers.Conv2DTranspose(
                        filters=64,
                        kernel_size=3,
                        strides=(2, 2),
                        padding="SAME",
                        activation=tf.nn.relu),
                tf.keras.layers.Conv2DTranspose(
                        filters=32,
                        kernel_size=3,
                        strides=(2, 2),
                        padding="SAME",
                        activation=tf.nn.relu),
                # No activation
                tf.keras.layers.Conv2DTranspose(
                        filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
            ]
        )
