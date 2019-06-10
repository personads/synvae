import numpy as np
import tensorflow as tf

class FFN:
    """FFN class (Position-wise Feed-Forward Networks)"""

    def __init__(self,
                 w1_dim=200,
                 w2_dim=100,
                 dropout=0.1):

        self.w1_dim = w1_dim
        self.w2_dim = w2_dim
        self.dropout = dropout

    def dense_relu_dense(self, inputs):
        output = tf.layers.dense(inputs, self.w1_dim, activation=tf.nn.relu)
        output =tf.layers.dense(output, self.w2_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def conv_relu_conv(self):
        raise NotImplementedError("i will implement it!")


def positional_encoding(dim, sentence_length, dtype=tf.float32):

    encoded_vec = np.array([pos/np.power(10000, 2*i/dim) for pos in range(sentence_length) for i in range(dim)])
    encoded_vec[::2] = np.sin(encoded_vec[::2])
    encoded_vec[1::2] = np.cos(encoded_vec[1::2])

    return tf.convert_to_tensor(encoded_vec.reshape([sentence_length, dim]), dtype=dtype)


class Attention:
    """Attention class"""

    def __init__(self,
                 num_heads=1,
                 masked=False,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=100,
                 dropout=0.2):

        assert linear_key_dim % num_heads == 0
        assert linear_value_dim % num_heads == 0

        self.num_heads = num_heads
        self.masked = masked
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.dropout = dropout

    def multi_head(self, q, k, v):
        q, k, v = self._linear_projection(q, k, v)
        qs, ks, vs = self._split_heads(q, k, v)
        outputs = self._scaled_dot_product(qs, ks, vs)
        output = self._concat_heads(outputs)
        output = tf.layers.dense(output, self.model_dim)

        return tf.nn.dropout(output, 1.0 - self.dropout)

    def _linear_projection(self, q, k, v):
        q = tf.layers.dense(q, self.linear_key_dim, use_bias=False)
        k = tf.layers.dense(k, self.linear_key_dim, use_bias=False)
        v = tf.layers.dense(v, self.linear_value_dim, use_bias=False)
        return q, k, v

    def _split_heads(self, q, k, v):

        def split_last_dimension_then_transpose(tensor, num_heads, dim):
            t_shape = tensor.get_shape().as_list()
            tensor = tf.reshape(tensor, [-1] + t_shape[1:-1] + [num_heads, dim // num_heads])
            return tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, num_heads, max_seq_len, dim]

        qs = split_last_dimension_then_transpose(q, self.num_heads, self.linear_key_dim)
        ks = split_last_dimension_then_transpose(k, self.num_heads, self.linear_key_dim)
        vs = split_last_dimension_then_transpose(v, self.num_heads, self.linear_value_dim)

        return qs, ks, vs

    def _scaled_dot_product(self, qs, ks, vs):
        key_dim_per_head = self.linear_key_dim // self.num_heads

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
            tensor = tf.transpose(tensor, [0, 2, 1, 3]) # [batch_size, max_seq_len, num_heads, dim]
            t_shape = tensor.get_shape().as_list()
            num_heads, dim = t_shape[-2:]
            return tf.reshape(tensor, [-1] + t_shape[1:-2] + [num_heads * dim])

        return transpose_then_concat_last_two_dimenstion(outputs)


class Encoder:
    """Encoder class"""

    def __init__(self,
                 num_layers=8,
                 num_heads=8,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=50,
                 ffn_dim=50,
                 dropout=0.2):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout

    def build(self, encoder_inputs):
        o1 = tf.identity(encoder_inputs)

        for i in range(1, self.num_layers+1):
            o2 = self._add_and_norm(o1, self._self_attention(q=o1,
                                                             k=o1,
                                                             v=o1), num=1)
            o3 = self._add_and_norm(o2, self._positional_feed_forward(o2), num=2)
            o1 = tf.identity(o3)

        return o3

    def _self_attention(self, q, k, v):
        attention = Attention(num_heads=self.num_heads,
                                masked=False,
                                linear_key_dim=self.linear_key_dim,
                                linear_value_dim=self.linear_value_dim,
                                model_dim=self.model_dim,
                                dropout=self.dropout)
        return attention.multi_head(q, k, v)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _positional_feed_forward(self, output):
        ffn = FFN(w1_dim=self.ffn_dim,
                  w2_dim=self.model_dim,
                  dropout=self.dropout)
        return ffn.dense_relu_dense(output)

class Decoder:
    """Decoder class"""

    def __init__(self,
                 num_layers=8,
                 num_heads=8,
                 linear_key_dim=50,
                 linear_value_dim=50,
                 model_dim=50,
                 ffn_dim=50,
                 dropout=0.2):

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.linear_key_dim = linear_key_dim
        self.linear_value_dim = linear_value_dim
        self.model_dim = model_dim
        self.ffn_dim = ffn_dim
        self.dropout = dropout

    def build(self, decoder_inputs, encoder_outputs):
        o1 = tf.identity(decoder_inputs)

        for i in range(1, self.num_layers+1):
            with tf.variable_scope('layer-%d' % i):
                o2 = self._add_and_norm(o1, self._masked_self_attention(q=o1,
                                                                        k=o1,
                                                                        v=o1), num=1)
                o3 = self._add_and_norm(o2, self._encoder_decoder_attention(q=o2,
                                                                            k=encoder_outputs,
                                                                            v=encoder_outputs), num=2)
                o4 = self._add_and_norm(o3, self._positional_feed_forward(o3), num=3)
                o1 = tf.identity(o4)

        return o4

    def _masked_self_attention(self, q, k, v):
        attention = Attention(num_heads=self.num_heads,
                                masked=True,  # Not implemented yet
                                linear_key_dim=self.linear_key_dim,
                                linear_value_dim=self.linear_value_dim,
                                model_dim=self.model_dim,
                                dropout=self.dropout)
        return attention.multi_head(q, k, v)

    def _add_and_norm(self, x, sub_layer_x, num=0):
        return tf.contrib.layers.layer_norm(tf.add(x, sub_layer_x)) # with Residual connection

    def _encoder_decoder_attention(self, q, k, v):
        attention = Attention(num_heads=self.num_heads,
                                masked=False,
                                linear_key_dim=self.linear_key_dim,
                                linear_value_dim=self.linear_value_dim,
                                model_dim=self.model_dim,
                                dropout=self.dropout)
        return attention.multi_head(q, k, v)

    def _positional_feed_forward(self, output):
        ffn = FFN(w1_dim=self.ffn_dim,
                  w2_dim=self.model_dim,
                  dropout=self.dropout)
        return ffn.dense_relu_dense(output)
