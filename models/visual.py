import logging, os, sys

import numpy as np
import PIL
import tensorflow as tf


class VisualVae:
    def __init__(self, img_height, img_width, img_depth, latent_dim, batch_size, learning_rate):
        self.img_height = img_height
        self.img_width = img_width
        self.img_depth = img_depth
        self.latent_dim = latent_dim
        self.batch_size = batch_size
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
        logging.info(self)


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
        logging.info("Saved model to '%s'." % save_path)


    def restore(self, tf_session, path):
        tf.train.Saver().restore(tf_session, path)
        logging.info("Restored model from '%s'." % path)


    def save_image(self, image_array, path):
        image = PIL.Image.fromarray((image_array * 255).astype(np.uint8))
        image.save(path)


    def train(self, tf_session, train_iter, valid_iter, max_epochs, model_path, out_path, tf_writer):
        merge_op = tf.summary.merge_all()
        next_op = train_iter.get_next()
        # initialize variables
        tf_session.run(tf.global_variables_initializer())
        # epoch training loop
        min_loss = None
        for epoch in range(1, max_epochs + 1):
            tf_session.run(train_iter.initializer)
            # iterate over batches
            avg_loss = 0.
            batch_idx = 0
            while True:
                try:
                    batch_idx += 1
                    batch = tf_session.run(next_op)
                    _, cur_loss, summaries = tf_session.run([self.train_op, self.loss, merge_op], feed_dict={self.images: batch})
                    avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                    sys.stdout.write("\repoch %d/%d. batch %d. avg_loss %.2f. cur_loss %.2f.   " % (epoch, max_epochs, batch_idx, avg_loss, cur_loss))
                    sys.stdout.flush()
                # end of dataset
                except tf.errors.OutOfRangeError:
                    # exit batch loop and proceed to next epoch
                    break
            # write epoch summary
            tf_writer.add_summary(summaries, epoch)
            logging.info("\rcompleted epoch %d/%d (%d batches). avg_loss %.2f.%s" % (epoch, max_epochs, batch_idx, avg_loss, ' '*32))

            # check performance on test split
            valid_loss = self.test(tf_session, valid_iter, epoch, out_path)
           
            # save latest model
            logging.info("saving latest model.")
            self.save(tf_session, os.path.join(model_path, 'latest_model.ckpt'))
            # check if model has improved
            if (min_loss is None) or (avg_loss < min_loss):
                logging.info("saving best model with avg_loss %.2f." % avg_loss)
                self.save(tf_session, os.path.join(model_path, 'best_model.ckpt'))
                min_loss = avg_loss


    def test(self, tf_session, iterator, epoch, out_path, export_step=5):
        next_op = iterator.get_next()
        tf_session.run(iterator.initializer)
        # iterate over batches
        avg_loss = 0.
        batch_idx = 0
        while True:
            try:
                batch_idx += 1
                sys.stdout.write("\revaluating batch %d." % (batch_idx))
                sys.stdout.flush()
                batch = tf_session.run(next_op)
                cur_loss, reconstructions = tf_session.run([self.loss, self.reconstructions], feed_dict={self.images: batch})
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                # save original image and reconstruction
                if (batch_idx-1) % export_step == 0:
                    if epoch == 1:
                        self.save_image(batch[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(epoch) + '_orig.png'))
                    self.save_image(reconstructions[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(epoch) + '_recon.png'))
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
        logging.info("\rcompleted evaluation (%d batches). avg_loss %.2f." % (batch_idx, avg_loss))
        return avg_loss



class CifarVae(VisualVae):
    def __init__(self, latent_dim, batch_size):
        super().__init__(img_height=32, img_width=32, img_depth=3, latent_dim=latent_dim, batch_size=batch_size, learning_rate=1e-4)


    def encode(self, images):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.img_height, self.img_width, self.img_depth))(images)
        conv1 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(inputs)
        conv2 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv1)
        conv3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(conv2)
        flat = tf.keras.layers.Flatten()(conv3)
        dense = tf.keras.layers.Dense(units=(self.img_height//4 *  self.img_width//4 * 64), activation=tf.nn.relu)(flat)
        mean = tf.keras.layers.Dense(self.latent_dim)(dense)
        logvar = tf.keras.layers.Dense(self.latent_dim)(dense)
        return mean, logvar


    def decode(self, latents):
        inputs = tf.keras.layers.InputLayer(input_shape=(self.latent_dim,))(latents)
        dense1 = tf.keras.layers.Dense(units=(self.img_height//4 *  self.img_width//4 * 32), activation=tf.nn.relu)(inputs)
        dense2 = tf.keras.layers.Dense(units=(self.img_height//4 *  self.img_width//4 * 64), activation=tf.nn.relu)(dense1)
        shape = tf.keras.layers.Reshape(target_shape=(self.img_height//4, self.img_width//4, 64))(dense2)
        deconv1 = tf.keras.layers.Conv2DTranspose(filters=128, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(shape)
        deconv2 = tf.keras.layers.Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation=tf.nn.relu)(deconv1)
        deconv3 = tf.keras.layers.Conv2DTranspose(filters=32, kernel_size=3, strides=1, padding='same', activation=tf.nn.relu)(deconv2)
        recons = tf.keras.layers.Conv2DTranspose(filters=self.img_depth, kernel_size=1, strides=1, padding="same", activation=tf.nn.sigmoid)(deconv3)
        return recons


class MnistVae(VisualVae):
    def __init__(self, latent_dim, batch_size):
        super().__init__(img_height=28, img_width=28, img_depth=1, latent_dim=latent_dim, batch_size=batch_size, learning_rate=1e-4)


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
