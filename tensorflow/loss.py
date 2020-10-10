# Wan, L., Wang, Q., Papir, A., Lopez Moreno, I., __Generalized End-to-End Loss for Speaker Verification__
# @see https://arxiv.org/abs/1710.10467
import tensorflow as tf
from scipy.spatial.distance import cosine


def similarity_matrix(embeddings, w, b):
    """
    Calculate the similarity matrix from a batch of embedding vectors. Eq. (9).

    In addition, we observed that removing e_{ji} when computing the
    centroid of the true speaker makes training stable and helps avoid
    trivial solutions. So, while we still use Equation 1 when calculating
    negative similarity (i.e., k 6 = j), we instead use Equation 8 when
    k = j
    """
    # batch_size, embedding_length = embeddings.shape

    # # compute the centroids
    # center =  # Eq (1)
    # center_excluding =  # Eq (8)

    # # generate the similarity matrix
    # S = None

    # S = tf.abs(w)*S+b
    # return S
    pass

def loss_from_similarity(similarity_matrix):
    pass

# this will need to accept the output from the final layer
def embedding_loss():
    mat = similarity_matrix()
    return loss_from_similarity(mat)
