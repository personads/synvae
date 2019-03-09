import copy, logging

import numpy as np
import tensorflow as tf

from magenta import music
from magenta.models.music_vae import configs

class MusicVae:
    """Wrapper class for a pre-trained MusicVAE model (based on magenta.models.music_vae.TrainedModel).
    """
    def __init__(self, config_name, batch_size):
        self.config_name = config_name
        self.batch_size = batch_size
        # load config
        self._config = copy.deepcopy(configs.CONFIG_MAP[config_name])
        # self._config.hparams.use_cudnn = tf.test.is_gpu_available() # enable cuDNN if available
        self.music_length = self._config.hparams.max_seq_len
        self.latent_dim = self._config.hparams.z_size
        # set up placeholders
        self.temperature = tf.placeholder(tf.float32, shape=(), name='temperature')
        if self._config.hparams.z_size:
            self._z_input = tf.placeholder(tf.float32, shape=[self.batch_size, self._config.hparams.z_size], name='aud_latents')
        else:
            self._z_input = None
        if self._config.data_converter.control_depth > 0:
            self._c_input = tf.placeholder(
            tf.float32, shape=[None, self._config.data_converter.control_depth])
        else:
            self._c_input = None
        self._inputs = tf.placeholder(tf.float32, shape=[self.batch_size, None, self._config.data_converter.input_depth])
        self._controls = tf.placeholder(tf.float32, shape=[self.batch_size, None, self._config.data_converter.control_depth])
        self.max_length = tf.constant(self.music_length, tf.int32)
        self._inputs_length = tf.placeholder(tf.int32, shape=[self.batch_size] + list(self._config.data_converter.length_shape))
        # set up encoding and decoding operation placeholders
        self.latents = None
        self.audios, self.lengths = None, None


    def __repr__(self):
        res  = '<MusicVae: '
        res += ' ♪ (%d steps)' % self.music_length
        res += ' -> ' + str(self.latent_dim)
        res += ' -> ♪ (%d steps)' % self.music_length
        res += '>'
        return res


    def build_core(self):
        # load model from config
        self._config.hparams.batch_size = self.batch_size
        self._config.data_converter.max_tensors_per_item = None
        self.model = self._config.model
        self.model.build(self._config.hparams, self._config.data_converter.output_depth, is_training=True)


    def build_encoder(self, audios, lengths):
        dist = self.model.encode(audios, lengths)
        latents = dist.sample()
        return latents


    def build_decoder(self, latents):
        audios, results = self.model.sample(self.batch_size, z=latents, max_length=self.max_length, temperature=self.temperature, c_input=self._c_input)
        lengths = results.final_sequence_lengths
        # if hierarchical, add up lengths of all n bars
        if len(lengths.shape) > 1:
            lengths = tf.reduce_sum(lengths, axis=1) # add up lengths of all n bars
        return audios, lengths


    def build(self):
        self.build_core()
        self.latents = self.build_encoder(self._inputs, self._inputs_length)
        self.audios, self.lengths = self.build_decoder(self._z_input)
        # debug info
        logging.info(self)


    def restore(self, tf_session, path, var_list=None):
        # tf.train.Saver(var_list=self.variables).restore(tf_session, path)
        tf.train.Saver(var_list=var_list).restore(tf_session, path)
        logging.info("[MusicVae] Restored model from '%s'." % path)


    def save_midi(self, audio_tensor, path):
        note_seq = self._config.data_converter.to_items([audio_tensor])[0]
        music.sequence_proto_to_midi_file(note_seq, path)


    def sample(self, tf_session, num_samples, temperature):
        feed_dict = {
            self._z_input: (np.random.randn(self.batch_size, self.latent_dim).astype(np.float32)),
            self.temperature: temperature
        }

        outputs = []
        for _ in range(int(np.ceil(num_samples / self.batch_size))):
            outputs.append(tf_session.run(self.audios, feed_dict))
        samples = np.vstack(outputs)[:num_samples]
        return samples
