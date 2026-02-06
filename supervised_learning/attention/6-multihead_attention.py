
#!/usr/bin/env python3
"""
6-multihead_attention.py
"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
    perform multi head attention
    """
    def __init__(self, dm, h):
        """
        dm is the dimensionality of the model
        h is the number of heads
        dm is divisible by h
         The multi head attention mechanism consists of four dense layers and a
         final linear layer. The first three dense layers are for the query, key,
         and value matrices, respectively. The fourth is for the output. The
         multi head attention mechanism splits the query, key, and value matrices
         into h heads, and concatenates the scaled dot product attention output of
         each head before passing it through the final linear layer
        """
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)

        self.linear = tf.keras.layers.Dense(dm)

    def call(self, Q, K, V, mask):
        """
        Q is a tensor of shape (batch, seq_len_q, dm) containing the input to
            generate the query matrix
        K is a tensor of shape (batch, seq_len_k, dm) containing the input to
            generate the key matrix
        V is a tensor of shape (batch, seq_len_v, dm) containing the input to
            generate the value matrix
        mask is always None
        The preceding three tensors are projected h times (one for each head)
        The multi head attention output is a tensor of shape (batch, seq_len_q, dm)
         containing the scaled dot product attention for each projected version
         of the query, key, and value matrices
         after they are concatenated
        """
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = tf.reshape(Q, (batch_size, -1, self.h, self.depth))
        K = tf.reshape(K, (batch_size, -1, self.h, self.depth))
        V = tf.reshape(V, (batch_size, -1, self.h, self.depth))

        Q = tf.transpose(Q, perm=[0, 2, 1, 3])
        K = tf.transpose(K, perm=[0, 2, 1, 3])
        V = tf.transpose(V, perm=[0, 2, 1, 3])

        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(
            scaled_attention, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, weights
