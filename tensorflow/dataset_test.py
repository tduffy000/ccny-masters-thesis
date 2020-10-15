import os
import yaml
import numpy as np

from dataset import GE2EDatasetLoader

config_file = 'conf/melspectrogram.yml'
with open(config_file, 'r') as stream:
    conf = yaml.safe_load(stream)
    stream.close()

data_conf = conf['feature_data']
dataset_loader = GE2EDatasetLoader(data_conf['path'])
N, M = data_conf['speakers_per_batch'], data_conf['utterances_per_speaker']

### HELPERS ###
def get_batch_labels(is_train):
    if is_train:
        _, labels = dataset_loader.get_single_train_batch()
    else:
        _, labels = dataset_loader.get_single_test_batch()
    return labels

def get_batch_speaker_cardinality(is_train):
    labels = get_batch_labels(is_train)
    return len(np.unique(labels))

def get_batch_utterance_counts(is_train):
    labels = get_batch_labels(is_train)
    return [ np.sum(labels == speaker) for speaker in np.unique(labels) ]

### TESTS ###
def test_train_dataset_batch_has_n_speakers():
    assert(get_batch_speaker_cardinality(True)) == N

def test_train_dataset_returns_m_utterances():
    counts = get_batch_utterance_counts(True)
    assert all([ c == M for c in counts ])

def test_test_dataset_batch_has_n_speakers():
    assert(get_batch_speaker_cardinality(False)) == N

def test_test_dataset_returns_m_utterances():
    counts = get_batch_utterance_counts(False)
    assert all([ c == M for c in counts ])