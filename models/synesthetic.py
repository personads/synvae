import logging, os, sys

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

class SynestheticVae:
    def __init__(self, visual_model, auditive_model, learning_rate):
        # load audio-visual models
        self.vis_model = visual_model
        self.aud_model = auditive_model
        self.latent_dim = self.aud_model.latent_dim
        self.learning_rate = learning_rate
        self.epoch = 0
        # set up computation graph placeholders
        self.images = self.vis_model.images
        self.temperature = self.aud_model.temperature


    def __repr__(self):
        res  = '<SynestheticVae: '
        res += str(self.images.shape[1:])
        res += ' -> ' + str(self.vis_latents.shape[1])
        res += ' -> ♪ (%d steps)' % self.aud_model.music_length
        res += ' -> ' + str(self.aud_latents.shape[1])
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res


    def calc_loss(self, originals, reconstructions, vis_dist, aud_dist):
        originals_flat = tf.reshape(originals, (-1, self.vis_model.img_height * self.vis_model.img_width * self.vis_model.img_depth))
        reconstructions_flat = tf.reshape(reconstructions, (-1, self.vis_model.img_height * self.vis_model.img_width * self.vis_model.img_depth))

        recon_loss = tf.reduce_sum(tf.square(originals_flat - reconstructions_flat), axis=1) # MSE

        prior_dist = tfp.distributions.MultivariateNormalDiag(loc=[0.] * self.latent_dim, scale_diag=[1.] * self.latent_dim)
        vis_latent_loss = tfp.distributions.kl_divergence(vis_dist, prior_dist)
        aud_latent_loss = tfp.distributions.kl_divergence(aud_dist, prior_dist)

        print("shapes before mean:", recon_loss.shape, vis_latent_loss.shape, aud_latent_loss.shape)
        recon_loss = tf.reduce_mean(recon_loss)
        vis_latent_loss = tf.reduce_mean(vis_latent_loss)
        aud_latent_loss = tf.reduce_mean(aud_latent_loss)
        print("shapes after mean:", recon_loss.shape, vis_latent_loss.shape, aud_latent_loss.shape)

        loss = recon_loss + vis_latent_loss + aud_latent_loss
        return loss


    def build_encoder(self, images):
        # visual encoding step
        with tf.variable_scope('visual_vae_encoder'):
            vis_latents, vis_dist = self.vis_model.build_encoder(images)
        # audio decoding step
        with tf.variable_scope('music_vae_decoder'):
            audios, lengths = self.aud_model.build_decoder(vis_latents)
        return audios, lengths, vis_latents, vis_dist


    def build_decoder(self, audios, lengths):
        # audio encoding step
        with tf.variable_scope('music_vae_encoder'):
            aud_latents, aud_dist = self.aud_model.build_encoder(audios, lengths)
        # visual decoding step
        with tf.variable_scope('visual_vae_decoder'):
            vis_recons = self.vis_model.build_decoder(aud_latents)
        return vis_recons, aud_latents, aud_dist


    def build(self):
        # set up core auditive model
        with tf.variable_scope('music_vae_core'):
            self.aud_model.build_core()
        # set up synesthetic encoder and decoder
        self.audios, self.aud_lengths, self.vis_latents, self.vis_dist = self.build_encoder(self.images)
        self.reconstructions, self.aud_latents, self.aud_dist = self.build_decoder(self.audios, self.aud_lengths)
        # only get variables relevant to visual vae
        self.train_variables = tf.trainable_variables(scope='visual_vae')
        self.fixed_variables = [var for var in tf.trainable_variables() if var not in self.train_variables]
        # set up loss calculation
        with tf.name_scope('loss'):
            self.loss = self.calc_loss(self.images, self.reconstructions, self.vis_dist, self.aud_dist)
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss)
        # debug info
        tf.summary.image('Originals', self.images, max_outputs=4)
        tf.summary.image('Reconstructions', self.reconstructions, max_outputs=4)
        tf.summary.scalar('Loss', self.loss)
        logging.info(self)


    def save(self, tf_session, path):
        save_path = tf.train.Saver().save(tf_session, path)
        logging.info("[SynestheticVae] Saved model to '%s'." % save_path)


    def restore(self, tf_session, path):
        tf.train.Saver().restore(tf_session, path)
        logging.info("[SynestheticVae] Restored model from '%s'." % path)


    def restore_auditive(self, tf_session, path):
        # remove music_vae scope prefix for loading pre-trained checkpoint
        var_map = {}
        for mvae_var in self.fixed_variables:
            # remove top level scope if present
            var_key = mvae_var.name
            if mvae_var.name.startswith('music_vae'):
                var_key = '/'.join(var_key.split('/')[1:])
            # remove device id
            var_key = var_key.split(':')[0]
            # map to actual variable
            var_map[var_key] = mvae_var
        self.aud_model.restore(tf_session=tf_session, path=path, var_list=var_map)


    def restore_visual(self, tf_session, path):
        # remove music_vae scope prefix for loading pre-trained checkpoint
        var_map = {}
        for vis_var in self.train_variables:
            # remove top level scope if present
            var_key = vis_var.name
            if vis_var.name.startswith('visual_vae'):
                var_key = '/'.join(var_key.split('/')[1:])
            # remove device id
            var_key = var_key.split(':')[0]
            # map to actual variable
            var_map[var_key] = vis_var
        self.vis_model.restore(tf_session=tf_session, path=path, var_list=var_map)


    def train(self, tf_session, train_iter, valid_iter, max_epochs, model_path, out_path, tf_writer):
        # set up training specific ops
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
                    temperature = 0.5
                    _, cur_loss, summaries, vis_latents, aud_latents = tf_session.run([self.train_op, self.loss, merge_op, self.vis_latents, self.aud_latents], feed_dict={self.images: batch, self.temperature: temperature})
                    # DBG latent difference
                    latent_diffs = np.absolute(vis_latents - aud_latents)
                    avg_latent_diff = np.mean(np.mean(latent_diffs, axis=1))
                    # END DBG
                    avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                    sys.stdout.write("\rEpoch %d/%d. Batch %d. avg_loss %.2f. cur_loss %.2f. avg_latent_diff %.4f.   " % (self.epoch, max_epochs, batch_idx, avg_loss, cur_loss, avg_latent_diff))
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
                temperature = 0.5
                cur_loss, audios, reconstructions = tf_session.run([self.loss, self.audios, self.reconstructions], feed_dict={self.images: batch, self.temperature: temperature})
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                # save original image and reconstruction
                if (export_step > 0) and ((batch_idx-1) % export_step == 0):
                    if self.epoch == 1:
                        self.vis_model.save_image(batch[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_orig.png'))
                    self.vis_model.save_image(reconstructions[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_recon.png'))
                    self.aud_model.save_midi(audios[0], os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_audio.mid'))
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
        logging.info("\rCompleted evaluation (%d batches). avg_loss %.2f." % (batch_idx, avg_loss))
        return avg_loss
