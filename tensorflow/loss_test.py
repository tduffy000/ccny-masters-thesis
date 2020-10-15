import numpy as np
from loss import get_embedding_loss

N, M = 3, 2
# [6,3]
# where the correct scores are:
# [[1, 0, 0],
#  [1, 0, 0],
#  [0, 1, 0],
#  [0, 1, 0],
#  [0, 0, 1],
#  [0, 0, 1]]

S_all_right = np.zeros((N*M, N))
for i in range(N):
    S_all_right[i*M:i*M+M,i] = 1.0

S_all_wrong = np.ones((N*M, N))
for i in range(N):
    S_all_wrong[i*M:i*M+M,i] = 0.0

### TESTS ###
def test_all_right_score():
    expected_score = 0.0
    l = get_embedding_loss(N, M)(None, S_all_right)
    assert l.numpy() == expected_score

def test_all_wrong_score():
    expected_score = N*N*M
    l = get_embedding_loss(N, M)(None, S_all_wrong)
    assert l.numpy() == expected_score