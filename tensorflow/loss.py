# Wan, L., Wang, Q., Papir, A., Lopez Moreno, I., __Generalized End-to-End Loss for Speaker Verification__
# @see https://arxiv.org/abs/1710.10467
import tensorflow as tf
import numpy as np

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
        # Eq (6) & Eq (10)
        return tf.math.reduce_sum(
            -S_true + tf.math.log(tf.math.reduce_sum(tf.exp(S), axis=1, keepdims=True) + 1e-6)
        )
    return loss

def false_acceptance_ratio(mat, N, M):
    """
    The ratio of falsely accepted impostor speakers over all scored impostors (type II errors).
    """
    pass

def false_rejection_ratio(mat, N, M):
    """
    The ratio of falsely rejected geniune speakers over all genuine speakers (type I errors).
    """
    pass

def equal_error_ratio(far, frr):
    """
    The ratio at which FAR and FRR (defined above) are equivalent.
    """
    return np.mean([far, frr])