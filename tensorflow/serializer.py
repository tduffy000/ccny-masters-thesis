import tensorflow as tf
import numpy as np

# TODO: add dataset field
class FeatureSerializer:

    def __init__(self, example_dim=3):
        assert(example_dim in [2,3])
        self.example_dim = example_dim
        self.speaker_id_mapping = {} # TODO: this will need to be dataset aware with inclusion of VoxCeleb

    @staticmethod
    def _int64_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    @staticmethod
    def _float_feature(value):
        if isinstance(value, np.ndarray) and value.ndim > 1:
            # only a natural conversion for 1-dimensional arrays
            value = value.reshape(-1)
        if not isinstance(value, list) and not isinstance(value, np.ndarray):
            value = [value]
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))

class SpectrogramSerializer(FeatureSerializer):

    def __init__(self):
        super().__init__()

    def serialize(self, batch, speaker_ids):
        for speaker_id in speaker_ids:
            if speaker_id not in self.speaker_id_mapping:
                self.speaker_id_mapping[speaker_id] = len(self.speaker_id_mapping)
        return tf.train.Example(features=tf.train.Features(feature={
            'feature/batch_size': self._int64_feature(batch.shape[0]),
            'feature/height': self._int64_feature(batch.shape[1]),
            'feature/width': self._int64_feature(batch.shape[2]),
            'spectrograms/encoded': self._float_feature(batch), # TODO: should this be a _float_feature or _bytes_feature???
            'speaker/orig_speaker_ids': self._int64_feature(speaker_ids)
        }))

    def deserialize(self, proto):
        feature_map = {
            'feature/batch_size': tf.io.FixedLenFeature([], tf.int64),
            'feature/height': tf.io.FixedLenFeature([], tf.int64),
            'feature/width': tf.io.FixedLenFeature([], tf.int64),
            'spectrograms/encoded': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'speaker/orig_speaker_ids': tf.io.FixedLenSequenceFeature([], tf.int64, allow_missing=True)
        }
        example = tf.io.parse_example(proto, feature_map)
        batch_size, height, width = example['feature/batch_size'], example['feature/height'], example['feature/width']
        inputs = tf.reshape(example['spectrograms/encoded'], [batch_size, height, width])
        targets = example['speaker/orig_speaker_ids']
        return inputs, targets