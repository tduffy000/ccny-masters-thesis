import tensorflow as tf
import numpy as np

# TODO: add dataset field
class FeatureSerializer:

    def __init__(self, example_dim=3):
        assert(example_dim in [2,3])
        self.example_dim = example_dim

    @staticmethod
    def _int64_feature(value):
        if isinstance(value, bool):
            value = int(value)
        if not isinstance(value, list):
            value = [value]
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    @staticmethod
    def _bytes_feature(value):
        if isinstance(value, str):
            value = bytes(value, 'utf-8')
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

    def __init__(self, target_speaker_id=None):
        super().__init__()
        self.target_speaker_id = target_speaker_id
        self.other_speaker_ids = set()

    def serialize(self, feature, speaker_id):
        """
        Args:
            feature:
            speaker_id: 
        Returns:

        """
        if speaker_id != self.target_speaker_id:
            self.other_speaker_ids.add(speaker_id)
        
        return tf.train.Example(features=tf.train.Features(feature={
            'spectrogram/height': self._int64_feature(feature.shape[0]),
            'spectrogram/width': self._int64_feature(feature.shape[1]),
            'spectrogram/encoded': self._float_feature(feature),
            'speaker/id': self._bytes_feature(speaker_id),
            'speaker/is_target': self._int64_feature(speaker_id == self.target_speaker_id),
            'data/source': self._bytes_feature(speaker_id.split('/')[0]),
            'data/subset': self._bytes_feature(speaker_id.split('/')[1])
        }))

    def deserialize(self, proto):
        """
        Args:
            proto
        Returns:

        """
        feature_map = {
            'spectrogram/height': tf.io.FixedLenFeature([], tf.int64),
            'spectrogram/width': tf.io.FixedLenFeature([], tf.int64),
            'spectrogram/encoded': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'speaker/id': tf.io.FixedLenFeature([], tf.string),
            'speaker/is_target': tf.io.FixedLenFeature([], tf.int64),
        }
        example = tf.io.parse_example(proto, feature_map)
        height, width = example['spectrogram/height'], example['spectrogram/width']
        inputs = tf.reshape(example['spectrogram/encoded'], [height, width])
        targets = example['speaker/is_target']
        return inputs, targets