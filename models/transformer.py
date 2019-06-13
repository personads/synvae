"""Transformer Network Library
Based on the very readable implementation by https://github.com/DongjunLee/transformer-tensorflow"""

import numpy as np
import tensorflow as tf


def build_point_wise_network(inputs, layer1_dim, latent_dim):
    output = tf.keras.layers.Dense(layer1_dim, activation=tf.nn.relu)(inputs)
    output = tf.keras.layers.Dense(latent_dim)(output)
    return tf.nn.dropout(output, 0.8)


def positional_encoding(dim, sentence_length, dtype=tf.float32):
    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])
    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


class MultiHeadAttention:
    def __init__(self, num_heads=1, masked=False, key_dim=50, value_dim=50, latent_dim=100):
        self.num_heads = num_heads
        self.masked = masked
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.latent_dim = latent_dim


    def multi_head(self, q, k, v):
        q = tf.keras.layers.Dense(self.key_dim, use_bias=False)(q)
        k = tf.keras.layers.Dense(self.key_dim, use_bias=False)(k)
        v = tf.keras.layers.Dense(self.value_dim, use_bias=False)(v)

        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs)
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.latent_dim)

        return tf.nn.dropout(output, 0.8)


    def _split_heads(self, q, k, v):
        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.value_dim)

        return qs, ks, vs


    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.key_dim // self.num_heads

        o1 = tf.matmul(qs, ks, transpose_b=True)
        o2 = o1 / (key_dim_per_head**0.5)

        if self.masked:
            diag_vals = tf.ones_like(o2[0, 0, :, :]) # (batch_size, num_heads, query_dim, key_dim)
            tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense() # (q_dim, k_dim)
            masks = tf.tile(tf.reshape(tril, [1, 1] + tril.get_shape().as_list()),
                            [tf.shape(o2)[0], tf.shape(o2)[1], 1, 1])
            paddings = tf.ones_like(masks) * -1e9
            o2 = tf.where(tf.equal(masks, 0), paddings, o2)

        o3 = tf.nn.softmax(o2)
        return tf.matmul(o3, vs)


    def _concat_heads(self, outputs):
        def transpose_then_concat_last_two_dimenstion(tensor):
            tensor = tf.transpose(tensor, [0, 2, 1, 3]) # (batch_size, max_seq_len, num_heads, dim)
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimenstion(outputs)


class Encoder:
    def __init__(self, num_layers, num_heads, key_dim, value_dim, latent_dim, ffn_dim):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.latent_dim = latent_dim
        self.ffn_dim = ffn_dim


    def build(self, encoder_inputs):
        o1 = tf.identity(encoder_inputs)
        for i in range(1, self.num_layers+1):
            o2 = self._add_and_norm(o1, self._self_attention(queries=o1, keys=o1, values=o1))
            o3 = self._add_and_norm(o2, build_point_wise_network(o2, self.ffn_dim, self.latent_dim))
            o1 = tf.identity(o3)
        return o3


    def _self_attention(self, queries, keys, values):
        attention = MultiHeadAttention(num_heads=self.num_heads, masked=False, key_dim=self.key_dim, value_dim=self.value_dim, latent_dim=self.latent_dim)
        return attention.multi_head(queries, keys, values)


    def _add_and_norm(self, x, sub_layer_x):
        return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection


class Decoder:
    def __init__(self, num_layers, num_heads, key_dim, value_dim, latent_dim, ffn_dim):
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.latent_dim = latent_dim
        self.ffn_dim = ffn_dim


    def build(self, decoder_inputs, encoder_outputs):
        o1 = tf.identity(decoder_inputs)
        for i in range(1, self.num_layers+1):
            o2 = self._add_and_norm(o1, self._masked_self_attention(queries=o1, keys=o1, values=o1))
            o3 = self._add_and_norm(o2, self._encoder_decoder_attention(queries=o2, keys=encoder_outputs, values=encoder_outputs))
            o4 = self._add_and_norm(o3, build_point_wise_network(o3, self.ffn_dim, self.latent_dim))
            o1 = tf.identity(o4)
        return o4


    def _masked_self_attention(self, queries, keys, values):
        attention = MultiHeadAttention(num_heads=self.num_heads, masked=True, key_dim=self.key_dim, value_dim=self.value_dim, latent_dim=self.latent_dim)
        return attention.multi_head(queries, keys, values)


    def _add_and_norm(self, x, sub_layer_x):
        return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection


    def _encoder_decoder_attention(self, queries, keys, values):
        attention = MultiHeadAttention(num_heads=self.num_heads, masked=False, key_dim=self.key_dim, value_dim=self.value_dim, latent_dim=self.latent_dim)
        return attention.multi_head(queries, keys, values)
