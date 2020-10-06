import tensorflow as tf
import numpy as np

class FeatureSerializer:

    def __init__(self, example_dim=3):
        assert(example_dim in [2,3])
        self.example_dim = example_dim
        self.speaker_id_mapping = {}

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

    def __init__(self, example_dim=3):
        super().__init__(example_dim)

    def serialize(self, spectrogram, speaker_id, file):
        if speaker_id not in self.speaker_id_mapping:
            self.speaker_id_mapping[speaker_id] = len(self.speaker_id_mapping)
        return tf.train.Example(features=tf.train.Features(feature={
            'spectrogram/height': self._int64_feature(spectrogram.shape[0]),
            'spectrogram/width': self._int64_feature(spectrogram.shape[1]),
            'spectrogram/encoded': self._float_feature(spectrogram), # TODO: should this be a _float_feature or _bytes_feature???
            'speaker/orig_speaker_id': self._int64_feature(speaker_id),
            'speaker/speaker_id_index': self._int64_feature(self.speaker_id_mapping[speaker_id]),
            'data/file': self._bytes_feature(bytes(file, 'utf-8'))
        }))

    def deserialize(self, proto):
        feature_map = {
            'spectrogram/height': tf.io.FixedLenFeature([], tf.int64),
            'spectrogram/width': tf.io.FixedLenFeature([], tf.int64),
            'spectrogram/encoded': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'speaker/orig_speaker_id': tf.io.FixedLenFeature([], tf.int64),
            'speaker/speaker_id_index': tf.io.FixedLenFeature([], tf.int64),
            'data/file': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_example(proto, feature_map)
        height, width = example['spectrogram/height'], example['spectrogram/width']
        if self.example_dim == 2:
            inputs = tf.reshape(example['spectrogram/encoded'], [height, width])
        else: # 3
            inputs = tf.reshape(example['spectrogram/encoded'], [height, width, 1])
        targets = example['speaker/speaker_id_index']
        return inputs, targets

class RawWaveformSerializer(FeatureSerializer):

    def __init__(self, example_dim=3):
        super().__init__(example_dim)

    def serialize(self, wv, speaker_id, file):
        if speaker_id not in self.speaker_id_mapping:
            self.speaker_id_mapping[speaker_id] = len(self.speaker_id_mapping)
        channels = 1 if wv.ndim == 1 else wv.shape[1]
        return tf.train.Example(features=tf.train.Features(feature={
            'waveform/samples': self._int64_feature(wv.shape[0]),
            'waveform/channels': self._int64_feature(channels),
            'waveform/encoded': self._float_feature(wv), # TODO: should this be a _float_feature or _bytes_feature???
            'speaker/orig_speaker_id': self._int64_feature(speaker_id),
            'speaker/speaker_id_index': self._int64_feature(self.speaker_id_mapping[speaker_id]),
            'data/file': self._bytes_feature(bytes(file, 'utf-8'))
        }))

    def deserialize(self, proto):
        feature_map = {
            'waveform/samples': tf.io.FixedLenFeature([], tf.int64),
            'waveform/channels': tf.io.FixedLenFeature([], tf.int64),
            'waveform/encoded': tf.io.FixedLenSequenceFeature([], tf.float32, allow_missing=True),
            'speaker/orig_speaker_id': tf.io.FixedLenFeature([], tf.int64),
            'speaker/speaker_id_index': tf.io.FixedLenFeature([], tf.int64),
            'data/file': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_example(proto, feature_map)
        channels, samples = example['waveform/channels'], example['waveform/samples']
        # TODO: how to reshape these?
        if self.example_dim == 2:
            inputs = tf.reshape(example['spectrogram/encoded'], [height, width])
        else: # 3
            inputs = tf.reshape(example['spectrogram/encoded'], [height, width, 1])
        targets = example['speaker/speaker_id_index']
        return inputs, targets
