import logging, os, sys

import numpy as np
import tensorflow as tf

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
        self.epsilons = self.vis_model.epsilons
        self.temperature = self.aud_model.temperature
        self.audios, self.aud_lengths, self.vis_latents, self.vis_means, self.vis_logvars = None, None, None, None, None
        self.reconstructions, self.aud_latents = None, None
        self.train_variables = None
        self.loss = None
        self.optimizer = None
        self.train_op = None


    def __repr__(self):
        res  = '<SynestheticVae: '
        res += str(self.images.shape[1:])
        res += ' -> ' + str(self.vis_latents.shape[1])
        res += ' -> ♪'
        res += ' -> ' + str(self.aud_latents.shape[1])
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res


    def build_encoder(self, images, epsilons):
        # visual encoding step
        with tf.variable_scope('visual_vae_encoder'):
            vis_latents, vis_means, vis_logvars = self.vis_model.build_encoder(images, epsilons)
        # audio decoding step
        with tf.variable_scope('music_vae_decoder'):
            audios, lengths = self.aud_model.build_decoder(vis_latents)
        return audios, lengths, vis_latents, vis_means, vis_logvars


    def build_decoder(self, audios, lengths):
        # audio encoding step
        with tf.variable_scope('music_vae_encoder'):
            aud_latents = self.aud_model.build_encoder(audios, lengths)
        # visual decoding step
        with tf.variable_scope('visual_vae_decoder'):
            vis_recons = self.vis_model.build_decoder(aud_latents)
        return vis_recons, aud_latents


    def build(self):
        self.audios, self.aud_lengths, self.vis_latents, self.vis_means, self.vis_logvars = self.build_encoder(self.images, self.epsilons)
        self.reconstructions, self.aud_latents = self.build_decoder(self.audios, self.aud_lengths)
        # only get variables relevant to visual vae
        self.train_variables = tf.trainable_variables(scope='visual_vae')
        self.fixed_variables = tf.trainable_variables(scope='music_vae')
        # set up loss calculation
        self.loss = self.vis_model.calc_loss(self.images, self.reconstructions, self.vis_means, self.vis_logvars)
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss, var_list=self.train_variables)
        # debug info
        logging.info(self)


    def restore_auditive(self, tf_session, path):
        # remove music_vae scope prefix for loading pre-trained checkpoint
        var_map = {}
        for mvae_var in self.fixed_variables:
            # remove top level scope
            var_key = '/'.join(mvae_var.name.split('/')[1:])
            # remove device id
            var_key = var_key.split(':')[0]
            # map to actual variable
            var_map[var_key] = mvae_var
        self.aud_model.restore(tf_session=tf_session, path=path, var_list=var_map)


    def train(self, tf_session, train_iter, valid_iter, max_epochs, model_path, out_path):
        next_op = train_iter.get_next()
        valid_next_op = valid_iter.get_next()
        # initialize variables
        tf_session.run(tf.global_variables_initializer())
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
                    epsilons = np.random.normal(loc=0., scale=1., size=(batch.shape[0], self.latent_dim))
                    temperature = 0.5
                    _, cur_loss = tf_session.run([self.train_op, self.loss], feed_dict={self.images: batch, self.epsilons: epsilons, self.temperature: temperature})
                    avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                    sys.stdout.write("\rEpoch %d/%d. Batch %d. avg_loss %.2f. cur_loss %.2f.   " % (self.epoch, max_epochs, batch_idx, avg_loss, cur_loss))
                    sys.stdout.flush()
                # end of dataset
                except tf.errors.OutOfRangeError:
                    # exit batch loop and proceed to next epoch
                    break
            # write epoch summary
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


    def test(self, tf_session, iterator, next_op, out_path, export_step=20):
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
                epsilons = np.zeros((batch.shape[0], self.latent_dim))
                cur_loss, audios, reconstructions = tf_session.run([self.loss, self.audios, self.reconstructions], feed_dict={self.images: batch, self.epsilons: epsilons})
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                # save original image and reconstruction
                if (export_step > 0) and ((batch_idx-1) % export_step == 0):
                    if self.epoch == 1:
                        self.vis_model.save_image(batch[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_orig.png'))
                    self.vis_model.save_image(reconstructions[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_recon.png'))
                    self.aud_model.save_midi(audios[0], os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '.midi'))
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
        logging.info("\rCompleted evaluation (%d batches). avg_loss %.2f." % (batch_idx, avg_loss))
        return avg_loss
