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
        # init placeholders
        self.texts = tf.placeholder(tf.int64, [self.batch_size, self.max_length], name='texts')
        self.tf_epoch = tf.placeholder(tf.int32, [], name='epoch')


    def __repr__(self):
        res  = '<%s: ' % self.__class__.__name__
        res += str(self.texts.shape[1:])
        res += ' -> %d (β=%s)' % (self.latent_dim, str(self.beta))
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res
    

    def build(self):
        self.encoder_inputs = self.texts[:, 1:] # no start token
        self.decoder_inputs = self.texts[:, :-1] # no end token

        # build encoder components
        encoder_embeddings = self.build_embedding(self.encoder_inputs)
        self.latents, means, sigmas = self.build_encoder(encoder_embeddings) # (batch_size, max_length-1, latent_dim)
        self.means, self.sigmas = means, sigmas

        # build decoder components
        recon_logits = self.build_decoder(self.latents)
        self.reconstructions = tf.argmax(recon_logits, axis=-1) # (batch_size, max_length-1)

        # build training components
        self.loss = self.calc_loss(self.encoder_inputs, recon_logits, means, sigmas)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

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


    def build_embedding(self, inputs):
        # reuse embedding layer since encoder and decoder "language" are one and the same
        with tf.variable_scope('embedding', reuse=True):
            input_embeddings = tf.keras.layers.Embedding(self.vocab_size, self.latent_dim)(inputs)

        positional_encoded = transformer.positional_encoding(self.latent_dim, self.max_length-1)
        position_inputs = tf.tile(tf.range(0, self.max_length-1), [self.batch_size])
        position_inputs = tf.reshape(position_inputs, [self.batch_size, self.max_length-1]) # batch_size x [0, 1, 2, ..., n]

        positional_embeddings = tf.add(input_embeddings, tf.nn.embedding_lookup(positional_encoded, position_inputs))
        positional_embeddings = tf.nn.dropout(positional_embeddings, keep_prob=.8)

        return positional_embeddings


    def build_encoder(self, encoder_embeddings):
        pass


    def build_decoder(self, decoder_embeddings, latents):
        pass


    def run_train_step(self, tf_session, batch):
        _, loss, recon_loss, latent_loss, means, sigmas, bw = tf_session.run([self.train_op, self.loss, self.recon_loss, self.latent_loss, self.means, self.sigmas, self.bw], feed_dict={self.texts: batch, self.tf_epoch: self.epoch})
        losses = {'All': loss, 'SeqXE': recon_loss, 'KL': latent_loss, 'm': np.mean(means), 's': np.mean(sigmas), 'bw': bw}
        return losses, None


    def run_test_step(self, tf_session, batch, batch_idx, out_path, export_step=5):
        loss, recon_loss, latent_loss, reconstructions, means, sigmas = tf_session.run([self.loss, self.recon_loss, self.latent_loss, self.reconstructions, self.means, self.sigmas], feed_dict={self.texts: batch, self.tf_epoch: self.epoch})
        losses = {'All': loss, 'SeqXE': recon_loss, 'KL': latent_loss, 'means': np.mean(means), 'sigmas': np.mean(sigmas)}
        # save original texts and reconstructions
        if (export_step > 0) and ((batch_idx-1) % export_step == 0):
            recon_txt = self.convert_indices_to_texts(reconstructions[0:1])[0]
            export_text = '%d (%d): "%s"' % (self.epoch, len(recon_txt), ' '.join(recon_txt))
            if self.epoch == 1:
                orig_txt = self.convert_indices_to_texts(batch[0:1, 1:])[0]
                export_text = 'ORIG (%d): "%s"\n' % (len(orig_txt), ' '.join(orig_txt)) + export_text
            self.save_text(export_text, os.path.join(out_path, str(batch_idx) + '_recons.txt'))
        return losses


    def save_text(self, text, path):
        with open(path, 'a', encoding='utf8') as fop:
            fop.write(text + '\n')


    def convert_indices_to_texts(self, indices):
        texts = []
        for o in range(indices.shape[0]):
            cur_text = []
            for t in range(indices.shape[1]):
                if indices[o, t] == self.vocab_size-1:
                    break
                cur_text.append(self.idx_tkn_map[indices[o, t]])
            texts.append(cur_text)
        return texts


class SarcVae(TextualVae):
    def __init__(self, idx_tkn_map, max_length, latent_dim, beta, batch_size):
        super().__init__(idx_tkn_map=idx_tkn_map, max_length=max_length, latent_dim=latent_dim, beta=beta, batch_size=batch_size, learning_rate=1e-4, beta_half_life=10)


    def build_encoder(self, encoder_embeddings):
        encoder = transformer.Encoder(
                 num_layers=4,
                 num_heads=4,
                 linear_key_dim=32,
                 linear_value_dim=32,
                 model_dim=self.latent_dim,
                 ffn_dim=32,
                 dropout=0.2)
        encoder_outputs = encoder.build(encoder_embeddings)

        # build VAE components with reparametrization
        means = tf.keras.layers.Dense(self.latent_dim)(encoder_outputs)
        sigmas = tf.keras.layers.Dense(self.latent_dim)(encoder_outputs)
        epsilons = tf.random.normal([self.batch_size, self.max_length-1, self.latent_dim], mean=0., stddev=1.)
        latents = means + tf.exp(.5 * sigmas) * epsilons # (batch_size, max_length-1, latent_dim)

        return latents, means, sigmas


    def build_decoder(self, latents):
        with tf.variable_scope('decoder', reuse=tf.AUTO_REUSE):
            decoder = transformer.Decoder(
                     num_layers=4,
                     num_heads=4,
                     linear_key_dim=32,
                     linear_value_dim=32,
                     model_dim=self.latent_dim,
                     ffn_dim=32,
                     dropout=0.2)

            decoder_embeddings = self.build_embedding(tf.zeros_like(self.decoder_inputs)) # enforce information bottleneck
            decoder_outputs = decoder.build(decoder_embeddings, latents)
            recon_logits = tf.layers.dense(decoder_outputs, self.vocab_size) # (batch_size, max_length-1, vocab_size)
        return recon_logits
