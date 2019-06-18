import logging, os, sys

import numpy as np
import tensorflow as tf

from models import transformer
from models.base import BaseModel

class TextualVae(BaseModel):
    def __init__(self, idx_tkn_map, max_length, latent_dim, beta, batch_size, learning_rate, beta_half_life):
        self.idx_tkn_map = idx_tkn_map
        self.vocab_size = len(self.idx_tkn_map)
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.beta_half_life = beta_half_life
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = 0
        self._recon_loss_name = 'SeqXE'
        # init placeholders
        self.originals = tf.placeholder(tf.int32, [self.batch_size, self.max_length], name='texts')
        self.embeddings = tf.Variable(tf.random_uniform([self.vocab_size, self.latent_dim]), name='embeddings')
        # self.decoder_inputs = self.originals[:, :-1] # no end token
        self.tf_epoch = tf.placeholder(tf.int32, [], name='epoch')


    def __repr__(self):
        res  = '<%s: ' % self.__class__.__name__
        res += str(self.originals.shape[1:])
        res += ' -> %d (β=%s)' % (self.latent_dim, str(self.beta))
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res
    

    def build(self):
        # build encoder components
        self.latents, means, sigmas = self.build_encoder(self.originals) # (batch_size, max_length-1, latent_dim)
        self.means, self.sigmas = means, sigmas

        # build decoder components
        recon_logits = self.build_decoder(self.latents)
        self.reconstructions = recon_logits

        # build training components
        self.loss_op = self.calc_loss(self.originals, recon_logits, means, sigmas)
        self.loss = self.recon_loss + self.beta * self.latent_loss #  actual loss to export
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss_op)

        logging.info(self)


    def calc_loss(self, originals, recon_logits, means, sigmas):
        # Reconstruction loss over sequence
        padding_mask = tf.cast(tf.math.logical_not(tf.math.equal(originals, 0)), dtype=tf.float32)
        self.recon_loss = tf.reduce_sum(tf.contrib.seq2seq.sequence_loss(logits=recon_logits, targets=originals, weights=padding_mask, average_across_timesteps=False))

        # KL divergence
        self.latent_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1. + sigmas - tf.square(means) - tf.exp(sigmas), axis=-1)) # mean KL over latent dims
        beta_weight = tf.sigmoid(tf.cast(self.tf_epoch, dtype=tf.float32) - self.beta_half_life)
        self.bw = beta_weight
        loss = self.recon_loss + self.beta * beta_weight * self.latent_loss
        return loss


    def build_encoder(self, texts):
        pass


    def build_decoder(self, latents):
        pass


    def run_train_step(self, tf_session, batch):
        _, loss, recon_loss, latent_loss, means, sigmas = tf_session.run([self.train_op, self.loss, self.recon_loss, self.latent_loss, self.means, self.sigmas], feed_dict={self.originals: batch, self.tf_epoch: self.epoch})
        losses = {'All': loss, self._recon_loss_name: recon_loss, 'KL': latent_loss, 'm': np.mean(means), 's': np.mean(sigmas)}
        return losses, None


    def run_test_step(self, tf_session, batch, batch_idx, out_path, export_step=5):
        loss, recon_loss, latent_loss, reconstructions, means, sigmas = tf_session.run([self.loss, self.recon_loss, self.latent_loss, self.reconstructions, self.means, self.sigmas], feed_dict={self.originals: batch, self.tf_epoch: self.epoch})
        losses = {'All': loss, self._recon_loss_name: recon_loss, 'KL': latent_loss, 'means': np.mean(means), 'sigmas': np.mean(sigmas)}
        # save original texts and reconstructions
        if (export_step > 0) and ((batch_idx-1) % export_step == 0):
            self.export_results(batch, reconstructions, out_path, batch_idx)
        return losses


    def export_results(self, originals, reconstructions, out_path, batch_idx, epoch_idx=None):
        epoch_idx = epoch_idx if epoch_idx else self.epoch

        recon_txt = self.convert_logits_to_texts(reconstructions[0:1])[0]
        export_text = '%d (%d): "%s"' % (epoch_idx, len(recon_txt), ' '.join(recon_txt))
        if epoch_idx == 1:
            orig_txt = self.convert_indices_to_texts(originals[0:1])[0]
            export_text = 'ORIG (%d): "%s"\n' % (len(orig_txt), ' '.join(orig_txt)) + export_text

        self.save_text(export_text, os.path.join(out_path, str(batch_idx) + '_recons.txt'))


    def save_text(self, text, path):
        with open(path, 'a', encoding='utf8') as fop:
            fop.write(text + '\n')


    def convert_logits_to_texts(self, recon_logits):
        recon_idcs = np.argmax(recon_logits, axis=-1)
        return self.convert_indices_to_texts(recon_idcs)


    def convert_indices_to_texts(self, indices):
        texts = []
        for o in range(indices.shape[0]):
            cur_text = []
            for t in range(indices.shape[1]):
                if indices[o, t] == self.vocab_size-2:
                    continue
                if indices[o, t] == self.vocab_size-1:
                    break
                cur_text.append(self.idx_tkn_map[indices[o, t]])
            texts.append(cur_text)
        return texts


class SarcVae(TextualVae):
    def __init__(self, idx_tkn_map, max_length, latent_dim, beta, batch_size):
        super().__init__(idx_tkn_map=idx_tkn_map, max_length=max_length, latent_dim=latent_dim, beta=beta, batch_size=batch_size, learning_rate=1e-4, beta_half_life=20)


    def build_encoder(self, texts):
        encoder_embeddings = tf.nn.embedding_lookup(self.embeddings, texts)

        fw_cell = tf.contrib.rnn.GRUCell(num_units=1024)
        encoder_outputs, encoder_states = tf.nn.dynamic_rnn(fw_cell, encoder_embeddings, dtype=tf.float32)

        # build VAE components with reparametrization
        means = tf.keras.layers.Dense(self.latent_dim)(encoder_states)
        sigmas = tf.keras.layers.Dense(self.latent_dim)(encoder_states)
        epsilons = tf.random.normal([self.batch_size, self.latent_dim], mean=0., stddev=1.)
        latents = means + tf.exp(.5 * sigmas) * epsilons # (batch_size, max_length-1, latent_dim)

        return latents, means, sigmas


    def build_decoder(self, latents):
        cell = tf.contrib.rnn.GRUCell(num_units=self.latent_dim)

        start_tokens = tf.fill([self.batch_size], self.vocab_size - 2)
        end_token = self.vocab_size - 1
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(self.embeddings, start_tokens=start_tokens, end_token=end_token)

        decoder = tf.contrib.seq2seq.BasicDecoder(
                                            cell=cell,
                                            helper=helper,
                                            initial_state=latents,
                                            output_layer=tf.layers.Dense(self.vocab_size))
        outputs, states, lengths = tf.contrib.seq2seq.dynamic_decode(
                                                        decoder=decoder,
                                                        output_time_major=False,
                                                        impute_finished=True,
                                                        maximum_iterations=self.max_length)

        recon_logits = tf.pad(
                        outputs.rnn_output,
                        [[0, 0], # no padding on batch
                         [0, tf.maximum(self.max_length - tf.reduce_max(lengths), 0)], # pad missing length
                         [0, 0]], # no padding on vocabulary
                        constant_values=0,
                        mode='CONSTANT')

        return recon_logits
