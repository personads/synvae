import logging, os, sys

import numpy as np
import tensorflow as tf

from models import transformer
from models.base import BaseModel

class TextualVae(BaseModel):
    def __init__(self, idx_tkn_map, max_length, latent_dim, beta, batch_size, learning_rate):
        self.idx_tkn_map = idx_tkn_map
        self.vocab_size = len(self.idx_tkn_map)
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = 0
        self.training = False
        # set up computation graph
        self.texts = tf.placeholder(tf.int64, [self.batch_size, self.max_length], name='texts')


    def __repr__(self):
        res  = '<%s: ' % self.__class__.__name__
        res += str(self.texts.shape[1:])
        res += ' -> %d (β=%s)' % (self.latent_dim, str(self.beta))
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res


    def build_encoder(self, texts):
        pass


    def build_decoder(self, latents):
        pass


    def build(self):
        tar_inp = self.texts[:, :-1]
        tar_real = self.texts[:, 1:]
        enc_padding_mask, look_ahead_mask, dec_padding_mask = self.create_masks(self.texts, tar_inp)

        self.latents = self.build_encoder(self.texts, mask=enc_padding_mask)
        self.reconstructions = self.build_decoder(self.latents, tar_inp, look_ahead_mask, dec_padding_mask)
        # set up loss
        self.loss = self.calc_loss(tar_real, self.reconstructions)
        predictions = tf.cast(tf.argmax(self.reconstructions, axis=-1), dtype=tf.int64)
        self.accuracy = tf.reduce_mean(tf.keras.metrics.sparse_categorical_accuracy(tar_real, predictions))
        # set up optimizer
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        # set up training operation
        self.train_op = self.optimizer.minimize(self.loss)
        logging.info(self)


    def calc_loss(self, originals, reconstructions):
        padding_mask = tf.cast(tf.math.logical_not(tf.math.equal(originals, 0)), dtype=tf.float32)
        # self.recon_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=reconstructions, labels=originals) * padding_mask)
        self.recon_loss = tf.contrib.seq2seq.sequence_loss(logits=reconstructions, targets=originals, weights=padding_mask)
        # KL divergence
        # self.latent_loss = tf.reduce_mean(-0.5 * tf.reduce_sum(1. + sigmas - tf.square(means) - tf.exp(sigmas), axis=-1)) # mean KL over latent dims
        # loss = self.recon_loss + self.beta * self.latent_loss
        return self.recon_loss


    def create_padding_mask(self, seq):
        seq = tf.cast(tf.math.equal(seq, 0), tf.float32)
        # add extra dimensions so that we can add the padding
        # to the attention logits.
        return seq[:, tf.newaxis, tf.newaxis, :]  # (batch_size, 1, 1, seq_len)


    def create_look_ahead_mask(self, size):
        mask = 1 - tf.linalg.band_part(tf.ones((size, size)), -1, 0)
        return mask  # (seq_len, seq_len)


    def create_masks(self, inp, tar):
        # Encoder padding mask
        enc_padding_mask = self.create_padding_mask(inp)

        # Used in the 2nd attention block in the decoder.
        # This padding mask is used to mask the encoder outputs.
        dec_padding_mask = self.create_padding_mask(inp)

        # Used in the 1st attention block in the decoder.
        # It is used to pad and mask future tokens in the input received by 
        # the decoder.
        look_ahead_mask = self.create_look_ahead_mask(tf.shape(tar)[1])
        dec_target_padding_mask = self.create_padding_mask(tar)
        combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

        return enc_padding_mask, combined_mask, dec_padding_mask


    def run_train_step(self, tf_session, batch):
        self.training = True
        _, loss, acc = tf_session.run([self.train_op, self.loss, self.accuracy], feed_dict={self.texts: batch})
        losses = {'All': loss, 'Acc': acc}
        return losses, None


    def run_test_step(self, tf_session, batch, batch_idx, out_path, export_step=5):
        loss, reconstructions, acc = tf_session.run([self.loss, self.reconstructions, self.accuracy], feed_dict={self.texts: batch})
        losses = {'All': loss, 'Acc': acc}
        # save original image and reconstruction
        if (export_step > 0) and ((batch_idx-1) % export_step == 0):
            recon_txt = self.convert_indices_to_texts(self.convert_output_to_indices(reconstructions[0:1]))[0]
            export_text = '%d (%d): "%s"' % (self.epoch, len(recon_txt), ' '.join(recon_txt))
            if self.epoch == 1:
                orig_txt = self.convert_indices_to_texts(batch[0:1])[0]
                export_text = 'ORIG (%d): "%s"\n' % (len(orig_txt), ' '.join(orig_txt)) + export_text
            self.save_text(export_text, os.path.join(out_path, str(batch_idx) + '_recons.txt'))
        return losses


    def save_text(self, text, path):
        with open(path, 'a', encoding='utf8') as fop:
            fop.write(text + '\n')


    def convert_output_to_indices(self, outputs):
        return np.argmax(outputs, axis=-1).astype(int)


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
        super().__init__(idx_tkn_map=idx_tkn_map, max_length=max_length, latent_dim=latent_dim, beta=beta, batch_size=batch_size, learning_rate=1e-3)


    def build_encoder(self, texts, mask=None):
        encoder = transformer.Encoder(num_layers=4, d_model=self.latent_dim, num_heads=8, dff=1024, vocab_size=self.vocab_size)
        latents = encoder(texts, training=self.training, mask=mask)
        return latents


    def build_decoder(self, latents, texts, lah_mask=None, pad_mask=None):
        decoder = transformer.Decoder(num_layers=4, d_model=self.latent_dim, num_heads=8, dff=1024, vocab_size=self.vocab_size)
        dec_out, attn = decoder(texts, latents, training=self.training, look_ahead_mask=lah_mask, padding_mask=pad_mask)
        recons = tf.keras.layers.Dense(self.vocab_size)(dec_out)
        return recons
