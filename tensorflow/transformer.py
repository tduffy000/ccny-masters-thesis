"""
Responsible for performing the feature transformations on the raw audio files
and serializing them to TFRecords.
"""
from collections import deque
import json
import os
import shutil
import librosa
import numpy as np
import tensorflow as tf
from features import FeatureExtractor
from serializer import SpectrogramSerializer
from logger import logger

class FeatureLoader:
    """
    Loads the raw .flac files and is responsible for performing:
    |-- feature extraction (mel spectrograms)
    |-- serialization into protobuf
    |-- write into TFRecord shards
    """
    def __init__(
        self,
        source_dir,
        target_dir,
        url,
        window_length, # in seconds
        overlap_percent, # in percent
        frame_length, # in seconds
        hop_length, # in seconds
        buffer_flush_size, # in features
        n_fft=512,
        n_mels=40,
        sr=16000,
        use_preemphasis=True,
        preemphasis_coef=0.97,
        trim_silence=True,
        trim_top_db=30,
        normalization=None,
        window='hamming'
    ):
        self.source_dir = f'{source_dir}/LibriSpeech/{url}'
        self.target_dir = f'{target_dir}/{url}'
        shutil.rmtree(self.target_dir, ignore_errors=True)
        os.makedirs(self.target_dir)
        self.extractor = FeatureExtractor(
            window_length=window_length,
            overlap_percent=overlap_percent,
            frame_length=frame_length,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            sr=sr,
            use_preemphasis=use_preemphasis,
            preemphasis_coef=preemphasis_coef,
            trim_silence=trim_silence,
            normalization=normalization,
            window=window
        )
        self.sr = sr
        self.num_files_created = 0
        self.tf_serializer = SpectrogramSerializer()
        self.feature_buffer = deque()
        self.buffer_flush_size = buffer_flush_size
        self.shape = None

    @staticmethod
    def _validate_numeric(feature):
        """
        :param feature: The numpy array feature to be written out.
        :type feature: numpy.ndarray
        """
        nans_found = np.sum(np.isnan(feature))
        infs_found = np.sum(np.isinf(feature))
        if nans_found > 0 or infs_found > 0:
            logger.fatal('Found a NaN or Infinity!')
            raise ValueError('Features must have real-valued elements.')

    def load(self):
        """
        Loads the raw .flac files contained in the LibriSpeech url's raw directory
        and performs the feature engineering steps defined by the FeatureExtractor
        and then writes them into TFRecord shard files so they can be loaded as
        a tf.data.TFRecordDatset.
        """
        for root, dirs, files in os.walk(self.source_dir):
            if files and not dirs: # at an audio file location
                speaker_id = int(root.split("/")[-2])
                for f in filter(lambda x: x.endswith('.flac'), files): # iterate over audio files and make features
                    y, _ = librosa.load(f'{root}/{f}', sr=self.sr)
                    for feature in self.extractor.as_melspectrogram(y):
                        if self.shape is None: # just so we only compute once
                            self.shape = feature.shape
                        self._validate_numeric(feature) # validate we're not writing np.nan or np.inf
                        protobuf = self.tf_serializer.serialize(feature, int(speaker_id))
                        self.feature_buffer.append(protobuf)
                        # flush the buffer into a TFRecord file
                        if len(self.feature_buffer) == self.buffer_flush_size:
                            path = f'{self.target_dir}/shard_{self.num_files_created+1:05d}.tfrecords'
                            logger.info(f'Flushing feature buffer into: {path}')
                            with tf.io.TFRecordWriter(path) as writer:
                                while True:
                                    if len(self.feature_buffer) == 0:
                                        break
                                    example = self.feature_buffer.pop()
                                    writer.write(example.SerializeToString())
                            self.feature_buffer = deque()
                            self.num_files_created += 1

        metadata = {
            'speaker_id_mapping': self.tf_serializer.speaker_id_mapping,
            'shape': self.shape,
            'total_files': self.num_files_created,
            'examples_per_file': self.buffer_flush_size
        }
        logger.info(f'Finished creating features, with metadata: {metadata}')
        with open(f'{self.target_dir}/metadata.json', 'w') as stream:
            json.dump(metadata, stream)
        return metadata
