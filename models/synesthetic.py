import logging, os, sys

import numpy as np
import tensorflow as tf

from models.base import BaseModel

class SynestheticVae(BaseModel):
    def __init__(self, visual_model, auditive_model, learning_rate):
        # load audio-visual models
        self.vis_model = visual_model
        self.aud_model = auditive_model
        self.latent_dim = self.aud_model.latent_dim
        self.learning_rate = learning_rate
        self.epoch = 0
        self.export_step = 5
        # set up computation graph placeholders
        self.originals = self.vis_model.originals
        self.temperature = self.aud_model.temperature


    def __repr__(self):
        res  = '<SynestheticVae: '
        res += str(self.originals.shape[1:])
        res += ' -> %s (β=%s)' % (str(self.vis_latents.shape[1]), str(self.vis_model.beta))
        res += ' -> ♪ (%d steps)' % self.aud_model.music_length
        res += ' -> ' + str(self.aud_latents.shape[1])
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res


    def build_encoder(self, originals):
        # visual encoding step
        with tf.variable_scope('visual_vae_encoder'):
            vis_latents, vis_means, vis_sigmas = self.vis_model.build_encoder(originals)
            print("vis_latents:", vis_latents.shape)
        # audio decoding step
        with tf.variable_scope('music_vae_decoder'):
            audios, lengths = self.aud_model.build_decoder(vis_latents)
        return audios, lengths, vis_latents, vis_means, vis_sigmas


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
        self.audios, self.aud_lengths, self.vis_latents, self.vis_means, self.vis_sigmas = self.build_encoder(self.originals)
        self.reconstructions, self.aud_latents, self.aud_dist = self.build_decoder(self.audios, self.aud_lengths)
        # only get variables relevant to visual vae
        self.train_variables = tf.trainable_variables(scope='visual_vae')
        self.fixed_variables = [var for var in tf.trainable_variables() if var not in self.train_variables]
        # set up loss calculation
        with tf.name_scope('loss'):
            self.loss = self.vis_model.calc_loss(self.originals, self.reconstructions, self.vis_means, self.vis_sigmas)
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss, var_list=self.train_variables)
        # debug info
        tf.summary.image('Originals', self.originals, max_outputs=4)
        tf.summary.image('Reconstructions', self.reconstructions, max_outputs=4)
        tf.summary.scalar('Loss', self.loss)
        logging.info(self)


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


    def run_train_step(self, tf_session, batch):
        feed_dict = {
            self.originals: batch,
            self.temperature: 1.0
        }
        if hasattr(self.vis_model, 'tf_epoch'): feed_dict[self.vis_model.tf_epoch] = self.epoch

        _, loss, recon_loss, latent_loss, summaries = tf_session.run([self.train_op, self.loss, self.vis_model.recon_loss, self.vis_model.latent_loss, self.merge_op], feed_dict=feed_dict)

        if hasattr(self.vis_model, 'beta_hl'): loss = recon_loss + self.beta * latent_loss
        losses = {'All': loss, self.vis_model._recon_loss_name: recon_loss, 'KL': latent_loss}
        return losses, summaries


    def run_test_step(self, tf_session, batch, batch_idx, out_path):
        feed_dict = {
            self.originals: batch,
            self.temperature: 0.5
        }
        if hasattr(self.vis_model, 'tf_epoch'): feed_dict[self.vis_model.tf_epoch] = self.epoch
        
        loss, recon_loss, latent_loss, audios, reconstructions = tf_session.run([self.loss, self.vis_model.recon_loss, self.vis_model.latent_loss, self.audios, self.reconstructions], feed_dict=feed_dict)

        if hasattr(self.vis_model, 'beta_hl'): loss = recon_loss + self.beta * latent_loss
        losses = {'All': loss, self.vis_model._recon_loss_name: recon_loss, 'KL': latent_loss}
        # save original image and reconstruction
        if (self.export_step > 0) and ((batch_idx-1) % self.export_step == 0):
            self.vis_model.export_results(batch, reconstructions, out_path, batch_idx, epoch_idx=self.epoch)
            self.aud_model.save_midi(audios[0], os.path.join(out_path, str(batch_idx) + '_' + str(self.epoch) + '_audio.mid'))
        return losses
