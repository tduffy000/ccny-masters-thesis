"""
Responsible for performing the feature transformations on the raw audio files
and serializing them to TFRecords.
"""
from collections import deque
import json
import random
import os
import shutil
import librosa
import numpy as np
import tensorflow as tf
from features import FeatureExtractor
from serializer import SpectrogramSerializer
from logger import logger

AUDIO_FILETYPES = ['.flac', '.wav']

class FeatureLoader:
    """
    Loads the raw .flac files and is responsible for performing:
    |-- feature extraction (mel spectrograms)
    |-- serialization into protobuf
    |-- write into TFRecord shards
    """
    def __init__(
        self,
        root_dir,
        datasets,
        target_dir,
        window_length, # in seconds
        overlap_percent, # in percent
        frame_length, # in seconds
        hop_length, # in seconds
        buffer_flush_size, # in features
        n_fft=512,
        n_mels=40,
        sr=16000,
        trim_top_db=30,
        test_data_ratio=0.2
    ):
        self.root_dir = root_dir
        self.datasets = datasets
        self.target_dir = target_dir
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
            trim_top_db=trim_top_db
        )
        self.sr = sr
        self.num_files_created = 0
        self.tf_serializer = SpectrogramSerializer()
        self.feature_buffer = deque()
        self.buffer_flush_size = buffer_flush_size
        self.shape = None
        self.test_data_ratio = test_data_ratio

    @staticmethod
    def _validate_numeric(feature):
        """
        Checks whether the feature to be written has invalid numeric values.

        :param feature: The numpy array feature to be written out.
        :type feature: numpy.ndarray
        """
        nans_found = np.sum(np.isnan(feature))
        infs_found = np.sum(np.isinf(feature))
        if nans_found > 0 or infs_found > 0:
            logger.fatal('Found a NaN or Infinity!')
            raise ValueError('Features must have real-valued elements.')

    @staticmethod
    def _is_audio_file(fname):
        """
        Confirm whether an input file is an audio file.

        :param fname: The name of the file with type suffix to be checked.
        :type fname: str
        :return: bool
        """
        for ftype in AUDIO_FILETYPES:
            if fname.endswith(ftype):
                return True
        return False

    def _get_audio_files(self, source_dir):
        """
        Walks the source directory tree and pulls out .flac files, creating a
        mapping of speaker_id -> [speaker specific files].

        :param source_dir: The root directory containing audio files to walk.
        :type source_dir: str
        :return: {speaker_id -> [speaker specific files]}
        """
        speaker_files = {}
        for root, dirs, files in os.walk(source_dir):
            if files and not dirs:
                # how do you pull out a VoxCeleb ID?
                speaker_id = int(root.split("/")[-2]) # this is specific to LibriSpeech
                d = speaker_files.get(speaker_id, [])
                paths = [ f'{root}/{f}' for f in filter(self._is_audio_file, files) ]
                speaker_files[speaker_id] = d + paths
        return speaker_files

    @staticmethod
    def _train_test_split_files(speaker_files, test_ratio):
        """
        Takes a mapping of files: speaker_id -> [speaker specific files] and then
        partitions them into disjoint training and test sets with the same format.

        :param speaker_files: A mapping of speaker_id -> [files]
        :type speaker_files: {speaker_id -> [files]}
        :param test_ratio: The ratio (0, 1.0] of files to place in the test set.
        :type test_ratio: float
        :return: ({speaker_id -> [files]}, {speaker_id -> [files]})
        """
        if len(speaker_files) == 0:
            raise RuntimeError('There are no files populated, make sure to call _get_audio_files first')
        train_files, test_files = {}, {}
        # first we need to ensure the cardinality of speaker sets in both is equivalent
        # note that this is train & test split with stratification ON;
        # e.g. both have sets have equivalent class proportions (which may not be
        # what we want as default behavior)
        for speaker, files in speaker_files.items():
            random.shuffle(files)
            split_idx = int(len(files) * (1 - test_ratio))
            train_files[speaker] = files[:split_idx]
            test_files[speaker] = files[split_idx:]
        return train_files, test_files

    # think about how we can de-couple this from the context
    def _raw_to_features(self, speaker_files, is_train):
        """
        """
        # TODO: we need to ensure that the buffer is flushed at the end of this dataset too
        # otherwise we'll hit an overlap
        subdir = 'train' if is_train else 'test'
        os.makedirs(f'{self.target_dir}/{subdir}')
        num_files_created = 0
        for speaker, files in speaker_files.items():
            for f in files:
                y, _ = librosa.load(f, sr=self.sr) # is this same for LibriSpeech & VoxCeleb1?
                # TODO: this has to be generalized given we're building multiple inputs here
                for feature in self.extractor.as_melspectrogram(y):
                    if self.shape is None:
                        self.shape = feature.shape
                    self._validate_numeric(feature)
                    protobuf = self.tf_serializer.serialize(feature, int(speaker), f)
                    self.feature_buffer.append(protobuf)
                    if len(self.feature_buffer) == self.buffer_flush_size:
                        path = f'{self.target_dir}/{subdir}/{subdir}_shard_{num_files_created+1:05d}.tfrecords'
                        logger.info(f'Flushing feature buffer into: {path}')
                        with tf.io.TFRecordWriter(path) as writer:
                            while True:
                                if len(self.feature_buffer) == 0:
                                    break
                                example = self.feature_buffer.pop()
                                writer.write(example.SerializeToString())
                        self.feature_buffer = deque()
                        num_files_created += 1
        return num_files_created

    def load(self):
        """
        Loads the raw .flac files contained in the LibriSpeech url's raw directory
        and performs the feature engineering steps defined by the FeatureExtractor
        and then writes them into TFRecord shard files so they can be loaded as
        a tf.data.TFRecordDatset.
        """
        file_lists = {}
        for dataset, subsets in self.datasets.items():
            dataset_path = f'{self.root_dir}/{dataset}'
            for subset in subsets:
                subset_path = f'{dataset_path}/{subset}'
                speaker_files = self._get_audio_files(subset_path)
                for speaker, files in speaker_files.items():
                    file_list = file_lists.get(speaker, []) + files
                    file_lists[speaker] = file_list

        files_created = 0
        train, test = self._train_test_split_files(file_lists, test_ratio=self.test_data_ratio)
        files_created += self._raw_to_features(train, is_train=True)
        files_created += self._raw_to_features(test, is_train=False)

        metadata = {
            'speaker_id_mapping': self.tf_serializer.speaker_id_mapping,
            'shape': self.shape,
            'total_files': files_created,
            'examples_per_file': self.buffer_flush_size,
            'test_ratio': self.test_data_ratio,
            'datasets': self.datasets
        }
        logger.info(f'Finished creating features, with metadata: {metadata}')
        with open(f'{self.target_dir}/metadata.json', 'w') as stream:
            json.dump(metadata, stream)
        return metadata
