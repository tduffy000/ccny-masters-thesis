# Wan, L., Wang, Q., Papir, A., Lopez Moreno, I., __Generalized End-to-End Loss for Speaker Verification__
# @see https://arxiv.org/abs/1710.10467
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import cosine

def get_embedding_loss(N, M):
    """
    Args:
        N: The number of unique speakers in a batch.
        M: The number of utterances from each speaker in a batch.
    """
    
    # cosine similarity is 1.0 if from same speaker, otherwise 0
    S_true = np.zeros((N*M, N))
    for i in range(N):
        S_true[i*M:i*M+M,i] = 1.0

    def loss(_, S):
        # Eq (6)
        return tf.math.reduce_sum(
            -1 * S_true + tf.math.log(tf.math.reduce_sum(tf.exp(S), axis=1, keepdims=True) + 1e-6)
        )
    return loss