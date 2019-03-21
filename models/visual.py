import logging, os, sys

import numpy as np
import PIL
import tensorflow as tf
import tensorflow_probability as tfp


class VisualVae:
    def __init__(self, img_height, img_width, img_depth, latent_dim, batch_size, learning_rate):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = 0
        # set up computation graph
        self.images = tf.placeholder(tf.float32, [self.batch_size, self.img_height, self.img_width, self.img_depth], name='images')


    def __repr__(self):
        res  = '<VisualVae: '
        res += str(self.images.shape[1:])
        res += ' -> ' + str(self.latent_dim)
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
        self.latents, self.latent_dist = self.build_encoder(self.images)
        self.reconstructions = self.build_decoder(self.latents)
        # set up loss
        self.loss = self.calc_loss(self.images, self.reconstructions, self.latent_dist)
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss)
        # debug info
        tf.summary.image('Originals', self.images, max_outputs=4)
        tf.summary.image('Reconstructions', self.reconstructions, max_outputs=4)
        tf.summary.scalar('Loss', self.loss)
        logging.info(self)


    def calc_loss(self, originals, reconstructions, latent_dist):
        originals_flat = tf.reshape(originals, (-1, self.img_height * self.img_width * self.img_depth))
        reconstructions_flat = tf.reshape(reconstructions, (-1, self.img_height * self.img_width * self.img_depth))

        recon_loss = tf.reduce_sum(tf.square(originals_flat - reconstructions_flat), axis=1) # MSE

        prior_dist = tfp.distributions.MultivariateNormalDiag(loc=[0.] * self.latent_dim, scale_diag=[1.] * self.latent_dim)
        latent_loss = tfp.distributions.kl_divergence(latent_dist, prior_dist)

        loss = tf.reduce_mean(recon_loss + latent_loss)
        return loss


    def save(self, tf_session, path):
        save_path = tf.train.Saver().save(tf_session, path)
        logging.info("[VisualVae] Saved model to '%s'." % save_path)


    def restore(self, tf_session, path, var_list=None):
        tf.train.Saver(var_list=var_list).restore(tf_session, path)
        logging.info("[VisualVae] Restored model from '%s'." % path)


    def save_image(self, image_array, path):
        image = PIL.Image.fromarray((image_array * 255).astype(np.uint8))
        image.save(path)


    def train(self, tf_session, train_iter, valid_iter, max_epochs, model_path, out_path, tf_writer):
        merge_op = tf.summary.merge_all()
        next_op = train_iter.get_next()
        valid_next_op = valid_iter.get_next()
        # epoch training loop
        min_loss = None
        while self.epoch < max_epochs:
            self.epoch += 1
            tf_session.run(train_iter.initializer)
            # iterate over batches
            avg_loss = 0.
            batch_idx = 0
            while True:
                try:
                    batch = tf_session.run(next_op)
                    batch_idx += 1
                    _, cur_loss, summaries = tf_session.run([self.train_op, self.loss, merge_op], feed_dict={self.images: batch})
                    avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                    sys.stdout.write("\rEpoch %d/%d. Batch %d. avg_loss %.2f. cur_loss %.2f.   " % (self.epoch, max_epochs, batch_idx, avg_loss, cur_loss))
                    sys.stdout.flush()
                # end of dataset
                except tf.errors.OutOfRangeError:
                    # exit batch loop and proceed to next epoch
                    break
            # write epoch summary
            tf_writer.add_summary(summaries, self.epoch)
            logging.info("\rCompleted epoch %d/%d (%d batches). avg_loss %.2f.%s" % (self.epoch, max_epochs, batch_idx, avg_loss, ' '*32))

            # check performance on test split
            valid_loss = self.test(tf_session, valid_iter, valid_next_op, out_path)
           
            # save latest model
            logging.info("Saving latest model...")
            self.save(tf_session, os.path.join(model_path, 'latest_model.ckpt'))
            # check if model has improved
            if (min_loss is None) or (valid_loss < min_loss):
                logging.info("Saving best model with valid_loss %.2f..." % valid_loss)
                self.save(tf_session, os.path.join(model_path, 'best_model.ckpt'))
                min_loss = valid_loss


    def test(self, tf_session, iterator, next_op, out_path, export_step=5):
        tf_session.run(iterator.initializer)
        # iterate over batches
        avg_loss = 0.
        batch_idx = 0
        while True:
            try:
                sys.stdout.write("\rEvaluating batch %d..." % (batch_idx))
                sys.stdout.flush()
                batch = tf_session.run(next_op)
                batch_idx += 1
                cur_loss, reconstructions = tf_session.run([self.loss, self.reconstructions], feed_dict={self.images: batch})
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                # save original image and reconstruction
                if (export_step > 0) and ((batch_idx-1) % export_step == 0):
                    if self.epoch == 1:
                        self.save_image(batch[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_orig.png'))
                    self.save_image(reconstructions[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_recon.png'))
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
        logging.info("\rCompleted evaluation (%d batches). avg_loss %.2f." % (batch_idx, avg_loss))
        return avg_loss



class CifarVae(VisualVae):
    def __init__(self, latent_dim, batch_size):
        super().__init__(img_height=32, img_width=32, img_depth=3, latent_dim=latent_dim, batch_size=batch_size, learning_rate=1e-4)


    def build_encoder(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv2)
        flat = tf.keras.layers.Flatten()(conv3)
        dense = tf.keras.layers.Dense(units=(self.img_height//8 * self.img_width//8 * 256), activation=tf.nn.relu)(flat)
        means = tf.keras.layers.Dense(self.latent_dim)(dense)
        sigmas = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.softplus)(dense)
        latent_dist = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=sigmas)
        latents = latent_dist.sample()
        return latents, latent_dist


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
    def __init__(self, latent_dim, batch_size):
        super().__init__(img_height=28, img_width=28, img_depth=1, latent_dim=latent_dim, batch_size=batch_size, learning_rate=1e-4)


    def build_encoder(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2, 2), activation=tf.nn.relu)(conv1)
        flat = tf.keras.layers.Flatten()(conv2)
        means = tf.keras.layers.Dense(self.latent_dim)(flat)
        sigmas = tf.keras.layers.Dense(self.latent_dim, activation=tf.nn.softplus)(flat)
        latent_dist = tfp.distributions.MultivariateNormalDiag(loc=means, scale_diag=sigmas)
        latents = latent_dist.sample()
        return latents, latent_dist


    def build_decoder(self, latents):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))(latents)
        dense = tf.keras.layers.Dense(units=7*7*32, activation=tf.nn.relu)(inputs)
        shape = tf.keras.layers.Reshape(target_shape=(7, 7, 32))(dense)
        deconv1 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)(shape)
        deconv2 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=(2, 2), padding='same', activation=tf.nn.relu)(deconv1)
        recons = tf.keras.layers.Conv2DTranspose(filters=1, kernel_size=3, strides=(1, 1), padding='same', activation=tf.nn.sigmoid)(deconv2)
        return recons
