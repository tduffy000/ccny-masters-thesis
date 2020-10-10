# Wan, L., Wang, Q., Papir, A., Lopez Moreno, I., __Generalized End-to-End Loss for Speaker Verification__
# @see https://arxiv.org/abs/1710.10467
import tensorflow as tf
from scipy.spatial.distance import cosine

def get_embedding_loss(N, M):
    
    def loss(S, _):
        S_correct = tf.concat([S[i*M:(i+1)*M, i:(i+1)] for i in range(N)], axis=0)
        return -tf.math.reduce_sum(
            S_correct - tf.math.log(tf.math.reduce_sum(tf.exp(S), axis=1, keepdims=True) + 1e-6)
        )

    return loss

