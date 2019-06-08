import logging, os, sys

import numpy as np
import tensorflow as tf

from models import transformer
from models.base import BaseModel

class SarcVae(BaseModel):
    def __init__(self, idx_tkn_map, max_length, latent_dim, beta, batch_size, learning_rate=1e-4):
        self.idx_tkn_map = idx_tkn_map
        self.vocab_size = len(self.idx_tkn_map)
        self.max_length = max_length
        self.latent_dim = latent_dim
        self.beta = beta
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epoch = 0
        self.training = True


    def __repr__(self):
        res  = '<%s: ' % self.__class__.__name__
        res += str(self.texts.shape[1:])
        res += ' -> %d (β=%s)' % (self.latent_dim, str(self.beta))
        res += ' -> ' + str(self.reconstructions.shape[1:])
        res += '>'
        return res


    def _init_placeholder(self):
        self.texts = tf.placeholder(tf.int64, [self.batch_size, self.max_length], name='texts')

        self.encoder_inputs = self.texts[:, 1:]

        # start_tokens = tf.fill([self.batch_size, 1], self.vocab_size-2)

        if self.training:
            # # slice last pad token
            # target_slice_last_1 = tf.slice(self.targets, [0, 0],
            #         [self.batch_size, self.max_length-1])
            # self.decoder_inputs = tf.concat([start_tokens, target_slice_last_1], axis=1)
            self.decoder_inputs = self.texts[:, :-1]
        else:
            pad_tokens = tf.zeros([self.batch_size, self.max_length-1], dtype=tf.int32) # 0: PAD ID
            self.decoder_inputs = tf.concat([start_tokens, pad_tokens], axis=1)


    def build(self):
        self._init_placeholder()

        encoder_emb_inp = self.build_embedding(self.encoder_inputs)
        self.encoder_outputs = self.build_encoder(encoder_emb_inp)

        decoder_emb_inp = self.build_embedding(self.decoder_inputs)
        decoder_outputs = self.build_decoder(decoder_emb_inp, self.encoder_outputs)

        output = tf.layers.dense(decoder_outputs, self.vocab_size)
        self.train_predictions = tf.argmax(output[0], axis=1)

        if self.training:
            predictions = tf.argmax(output, axis=-1)
            output, predictions
        else:
            next_decoder_inputs = self._filled_next_token(self.decoder_inputs, output, 1)

            # predict output with loop. [encoder_outputs, decoder_inputs (filled next token)]
            for i in range(2, self.max_length):
                decoder_emb_inp = self.build_embedding(next_decoder_inputs, encoder=False, reuse=True)
                decoder_outputs = self.build_decoder(decoder_emb_inp, self.encoder_outputs, reuse=True)
                output = self.build_output(decoder_outputs, reuse=True)

                next_decoder_inputs = self._filled_next_token(next_decoder_inputs, output, i)

            # slice start_token
            decoder_input_start_1 = tf.slice(next_decoder_inputs, [0, 1],
                    [self.batch_size, self.max_length-1])
            predictions = tf.concat(
                    [decoder_input_start_1, tf.zeros([self.batch_size, 1], dtype=tf.int32)], axis=1)
            output, predictions

        self.reconstructions = predictions
        self.loss = self.calc_loss(output, self.encoder_inputs)
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)
        logging.info(self)


    def calc_loss(self, reconstructions, originals):
        padding_mask = tf.cast(tf.math.logical_not(tf.math.equal(originals, 0)), dtype=tf.float32)
        return tf.contrib.seq2seq.sequence_loss(logits=reconstructions, targets=originals, weights=padding_mask)


    def build_embedding(self, inputs):
        embeddings = tf.keras.layers.Embedding(self.vocab_size, self.latent_dim)(inputs)

        positional_encoded = transformer.positional_encoding(self.latent_dim, self.max_length-1)

        # Add
        position_inputs = tf.tile(tf.range(0, self.max_length-1), [self.batch_size])
        position_inputs = tf.reshape(position_inputs,
                                     [self.batch_size, self.max_length-1]) # batch_size x [0, 1, 2, ..., n]

        encoded_inputs = tf.add(embeddings,
                         tf.nn.embedding_lookup(positional_encoded, position_inputs))

        return tf.nn.dropout(encoded_inputs, keep_prob=.8)


    def build_encoder(self, encoder_emb_inp):
        encoder = transformer.Encoder(
                 linear_key_dim=self.latent_dim,
                 linear_value_dim=self.latent_dim,
                 model_dim=self.latent_dim)
        return encoder.build(encoder_emb_inp)


    def build_decoder(self, decoder_emb_inp, encoder_outputs):
        decoder = transformer.Decoder(
                 linear_key_dim=self.latent_dim,
                 linear_value_dim=self.latent_dim,
                 model_dim=self.latent_dim)

        return decoder.build(decoder_emb_inp, encoder_outputs)


    def _filled_next_token(self, inputs, logits, decoder_index):
        tf.argmax(logits[0], axis=1, output_type=tf.int32)

        next_token = tf.slice(
                tf.argmax(logits, axis=2, output_type=tf.int32),
                [0, decoder_index-1],
                [self.batch_size, 1])
        left_zero_pads = tf.zeros([self.batch_size, decoder_index], dtype=tf.int32)
        right_zero_pads = tf.zeros([self.batch_size, (self.max_length-1)], dtype=tf.int32)
        next_token = tf.concat((left_zero_pads, next_token, right_zero_pads), axis=1)

        return inputs + next_token


    def run_train_step(self, tf_session, batch):
        self.training = True
        _, loss = tf_session.run([self.train_op, self.loss], feed_dict={self.texts: batch})
        losses = {'All': loss}
        return losses, None


    def run_test_step(self, tf_session, batch, batch_idx, out_path, export_step=5):
        loss, reconstructions = tf_session.run([self.loss, self.reconstructions], feed_dict={self.texts: batch})
        losses = {'All': loss}
        # save original image and reconstruction
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
