import tensorflow as tf
import numpy as np

def get_embedding_loss(N, M):
    """
    Get the loss function relating to the Generalized End-to-End model architecture which 
    returns a cosine similarity matrix of each speaker's embedding to the speaker centroids. 

    Args:
        N: The number of unique speakers in a batch.
        M: The number of utterances from each speaker in a batch.
    Returns:
        A closure loss function taking in the similarity matrix generated by model.SpeakerSimilarityMatrixLayer.
    """
    def loss(_, S):
        # Eq (6) & Eq (10)
        S_correct = tf.concat([S[i*M:(i+1)*M, i:(i+1)] for i in range(N)], axis=0)

        l = tf.math.reduce_sum(
            -S_correct + tf.math.log(tf.math.reduce_sum(tf.exp(S), axis=1, keepdims=True) + 1e-6)
        )
        return l
    return loss

def false_acceptance_ratio(S_thres, N, M):
    """
    S_thres [NM x N]
    The ratio of falsely accepted impostor speakers over all scored impostors (type II errors).
    """
    total_fp = sum([ np.sum(S_thres[i,:]) - np.sum(S_thres[i,i:i+M]) for i in range(N)])
    return total_fp/((N-1)*M)

def false_rejection_ratio(S_thres, N, M):
    """
    The ratio of falsely rejected geniune speakers over all genuine speakers (type I errors).
    """
    total_fn = sum([M - np.sum(S_thres[i,i:i+M]) for i in range(N)])
    return total_fn/(N*M)

def equal_error_ratio(S, N, M, threshold_start=0.5, threshold_step=0.01, iters=50):
    """
    The ratio at which FAR and FRR (defined above) are equivalent.
    """
    epsilon, eer, eer_thres = 1.0, -1.0, -1.0
    for threshold in [threshold_step*i + threshold_start for i in range(iters)]:
        S_thres = S > threshold
        far = false_acceptance_ratio(S_thres, N, M)
        frr = false_rejection_ratio(S_thres, N, M)
        if abs(far-frr) < epsilon:
            epsilon = abs(far-frr)
            eer_thres = threshold
            eer = (far+frr)/2
    return threshold, eer