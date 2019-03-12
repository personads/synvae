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
        self.audios, self.aud_dists, self.aud_lengths = None, None, None
        self.vis_latents, self.vis_means, self.vis_vars = None, None, None
        self.reconstructions, self.aud_latents = None, None
        self.train_variables = None
        self.loss = None
        self.optimizer = None
        self.train_op = None


    def __repr__(self):
        res  = '<SynestheticVae: '
        res += str(self.images.shape[1:])
        res += ' -> ' + str(self.vis_latents.shape[1])
        res += ' -> ♪ (%d steps)' % self.aud_model.music_length
        res += ' -> ' + str(self.aud_latents.shape[1])
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res


    def build_encoder(self, images, epsilons):
        # visual encoding step
        with tf.variable_scope('visual_vae_encoder'):
            vis_latents, vis_means, vis_vars = self.vis_model.build_encoder(images, epsilons)
        # audio decoding step
        with tf.variable_scope('music_vae_decoder'):
            audios, aud_dists, lengths = self.aud_model.build_decoder(vis_latents)
        return audios, aud_dists, lengths, vis_latents, vis_means, vis_vars


    def build_decoder(self, aud_dists, lengths, epsilons):
        # audio encoding step
        with tf.variable_scope('music_vae_encoder'):
            aud_latents = self.aud_model.build_encoder(aud_dists, lengths, epsilons)
        # visual decoding step
        with tf.variable_scope('visual_vae_decoder'):
            vis_recons = self.vis_model.build_decoder(aud_latents)
        return vis_recons, aud_latents


    def build(self):
        # set up core auditive model
        with tf.variable_scope('music_vae_core'):
            self.aud_model.build_core()
        # set up synesthetic encoder and decoder
        self.audios, self.aud_dists, self.aud_lengths, self.vis_latents, self.vis_means, self.vis_vars = self.build_encoder(self.images, self.epsilons)
        self.reconstructions, self.aud_latents = self.build_decoder(self.aud_dists, self.aud_lengths, self.epsilons)
        # only get variables relevant to visual vae
        self.train_variables = tf.trainable_variables(scope='visual_vae')
        # self.fixed_variables = tf.trainable_variables(scope='music_vae')
        self.fixed_variables = [var for var in tf.trainable_variables() if var not in self.train_variables]
        # set up loss calculation
        self.loss = self.vis_model.calc_loss(self.images, self.reconstructions, self.vis_means, self.vis_vars)
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss, var_list=self.train_variables)
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
        # # DBG Print variables
        # ckpt_vars = sorted([name for name, _ in tf.train.list_variables(path)])
        # model_vars = sorted(list(var_map.keys()))
        # global_vars = sorted([v.name for v in tf.global_variables()])
        # print("---- checkpoint variables ----")
        # print("\n".join(ckpt_vars))
        # print("---- music variables ----")
        # print("\n".join(model_vars))
        # print("---- global variables ----")
        # print("\n".join(global_vars))
        # # END DBG
        self.aud_model.restore(tf_session=tf_session, path=path, var_list=var_map)


    # DEBUG function
    def export_weights(self, tf_session, path):
        with open(path, 'w', encoding='utf8') as fop:
            # export and save weights
            # check fixed weights
            fop.write('-'*16 + 'FIXED' + '-'*16 + '\n')
            var_vals = tf_session.run(self.fixed_variables)
            for var, val in zip(self.fixed_variables, var_vals):
                self.weight_vals[var.name] = self.weight_vals.get(var.name, []) + [val]
                # check integrity
                update_state = 'init'
                if len(self.weight_vals[var.name]) > 1:
                    update_state = 'changed'
                    if np.array_equal(self.weight_vals[var.name][-1], self.weight_vals[var.name][-2]):
                        update_state = 'unchanged'
                # write to file
                fop.write('(' + update_state + ') ' + var.name + ': ' + str(val) + '\n')
            # check trainable variables
            fop.write('-'*16 + 'TRAINABLE' + '-'*16 + '\n')
            var_vals = tf_session.run(self.train_variables)
            for var, val in zip(self.train_variables, var_vals):
                self.weight_vals[var.name] = self.weight_vals.get(var.name, []) + [val]
                # check integrity
                update_state = 'init'
                if len(self.weight_vals[var.name]) > 1:
                    update_state = 'changed'
                    if np.array_equal(self.weight_vals[var.name][-1], self.weight_vals[var.name][-2]):
                        update_state = 'unchanged'
                # write to file
                fop.write('(' + update_state + ') ' + var.name + ': ' + str(val) + '\n')
    # END DEBUG function


    def train(self, tf_session, train_iter, valid_iter, max_epochs, model_path, out_path, tf_writer):
        # DEBUG
        # self.weight_vals = {}
        # END DEBUG
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
                    epsilons = np.random.normal(loc=0., scale=1., size=(batch.shape[0], self.latent_dim))
                    temperature = 0.5
                    _, cur_loss, summaries, vis_latents, aud_latents = tf_session.run([self.train_op, self.loss, merge_op, self.vis_latents, self.aud_latents], feed_dict={self.images: batch, self.epsilons: epsilons, self.temperature: temperature})
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
                epsilons = np.zeros((batch.shape[0], self.latent_dim))
                temperature = 0.5
                cur_loss, audios, reconstructions = tf_session.run([self.loss, self.audios, self.reconstructions], feed_dict={self.images: batch, self.epsilons: epsilons, self.temperature: temperature})
                avg_loss = ((avg_loss * (batch_idx - 1)) + cur_loss) / batch_idx
                # save original image and reconstruction
                if (export_step > 0) and ((batch_idx-1) % export_step == 0):
                    if self.epoch == 1:
                        self.vis_model.save_image(batch[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_orig.png'))
                    self.vis_model.save_image(reconstructions[0].squeeze(), os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_recon.png'))
                    self.aud_model.save_midi(audios[0], os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '.mid'))
            # end of dataset
            except tf.errors.OutOfRangeError:
                # exit batch loop and proceed to next epoch
                break
        logging.info("\rCompleted evaluation (%d batches). avg_loss %.2f." % (batch_idx, avg_loss))
        return avg_loss
