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
from features import SpectrogramExtractor
from serializer import SpectrogramSerializer
from logger import logger

AUDIO_FILETYPES = ['.flac', '.wav']
FILE_THRESHOLD = 5

class GE2EBatchLoader:
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
        train_split,
        target_dir,
        window_length, # in seconds
        overlap_percent, # in percent
        frame_length, # in seconds
        hop_length, # in seconds
        target_speaker_id,
        n_fft=512,
        n_mels=40,
        sr=16000,
        trim_top_db=30,
        tfrecord_examples_per_file=100
    ):
        self.root_dir = root_dir
        self.datasets = datasets
        self.target_dir = target_dir
        shutil.rmtree(self.target_dir, ignore_errors=True)
        os.makedirs(self.target_dir)
        self.extractor = SpectrogramExtractor(
            window_length=window_length,
            overlap_percent=overlap_percent,
            frame_length=frame_length,
            hop_length=hop_length,
            n_fft=n_fft,
            n_mels=n_mels,
            sr=sr,
            trim_top_db=trim_top_db
        )
        self.tf_serializer = SpectrogramSerializer(target_speaker_id)
        self.sr = sr
        self.num_files_created = {}
        self.target_speaker_id = target_speaker_id

        self.shape = None
        self.train_split = train_split
        self.feature_buffer = deque()
        self.buffer_flush_size = tfrecord_examples_per_file

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

    def _get_speaker_id_from_path(self, path, parent):
        if parent == 'LibriSpeech':
            return path.split("/")[-2]

    def _get_audio_files(self, root_dir, datasets):
        speaker_files = {}
        for parent, subsets in datasets.items():
            for subset in subsets:
                path = f'{root_dir}/{parent}/{subset}'
                for root, dirs, files in os.walk(path):
                    if files and not dirs:
                        # TODO: provide helper for dealing with non-LibriSpeech id
                        speaker_id = f'{parent}/{subset}/{int(root.split("/")[-2])}'
                        d = speaker_files.get(speaker_id, [])
                        paths = [ f'{root}/{f}' for f in filter(self._is_audio_file, files) ]
                        speaker_files[speaker_id] = d + paths
        return speaker_files

    @staticmethod
    def _train_test_split_files(speaker_files, train_split):
        assert(train_split > 0 and train_split < 1), "train_split must be (0, 1]!"
        test_split = 1 - train_split
        train_files = {}
        test_files = {}
        for speaker, files in speaker_files.items():
            total_files = len(files)
            train_idx = random.sample(range(total_files), k=int(total_files * train_split))
            for idx, f in enumerate(files):
                if idx in train_idx:
                    train_files[speaker] = train_files.get(speaker, []) + [f]
                else:
                    test_files[speaker] = test_files.get(speaker, []) + [f]

        return train_files, test_files

    def _load_dataset(self, subdir, speaker_files):
        target_root = f'{self.target_dir}/{subdir}'        
        os.makedirs(target_root)
        num_files_created = 0
        for speaker, files in speaker_files.items():
            for f in files:
                y, _ = librosa.load(f, sr=self.sr)
                for feature in self.extractor.as_features(y):
                    if self.shape is None:
                        self.shape = feature.shape
                    self._validate_numeric(feature)
                    protobuf = self.tf_serializer.serialize(feature, speaker)
                    self.feature_buffer.append(protobuf)
                    if len(self.feature_buffer) == self.buffer_flush_size:
                        path = f'{target_root}/{subdir}_shard_{num_files_created+1:05d}.tfrecords'
                        logger.info(f'Flushing feature buffer into: {path}')
                        with tf.io.TFRecordWriter(path) as writer:
                            while len(self.feature_buffer) > 0:
                                example = self.feature_buffer.pop()
                                writer.write(example.SerializeToString())
                            self.feature_buffer = deque()
                            num_files_created += 1
        # reset the feature buffer so we don't leak into test set
        self.num_files_created[subdir] = num_files_created
        self.feature_buffer = deque()

    def load(self):

        # partition dataset into test and train partitions
        # MUST BE STRATIFIED BY SPEAKER ID proportions
        all_speaker_files = self._get_audio_files(self.root_dir, self.datasets)
        train_files, test_files = self._train_test_split_files(all_speaker_files, self.train_split)

        # load & write out train dataset
        self._load_dataset('train', train_files)

        # load & write out test dataset
        self._load_dataset('test', test_files)

        metadata = {
            'feature_shape': self.shape,
            'files_created': self.num_files_created,
            'target_speaker_id': self.target_speaker_id,
            'datasets': self.datasets,
            'train_test_split': [self.train_split, 1 - self.train_split]
        }
        logger.info(f'Finished creating features, with metadata: {metadata}')
        with open(f'{self.target_dir}/metadata.json', 'w') as stream:
            json.dump(metadata, stream)
        return metadata