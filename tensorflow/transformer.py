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
from serializer import SpectrogramSerializer, GE2ESpectrogramSerializer
from logger import logger

AUDIO_FILETYPES = ['.flac', '.wav']
FILE_THRESHOLD = 5

class BatchLoader:
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
        self.tf_serializer = SpectrogramSerializer()
        self.sr = sr
        self.num_files_created = {}
        self.speaker_id_map = {}

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

                        # update our speaker_id mapping
                        if speaker_id not in self.speaker_id_map:
                            self.speaker_id_map[speaker_id] = len(self.speaker_id_map)
                        
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

                    mapped_id = self.speaker_id_map[speaker]
                    protobuf = self.tf_serializer.serialize(feature, speaker, mapped_id)
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

        self.num_files_created[subdir] = num_files_created
        self.feature_buffer = deque()

    def load(self):

        all_speaker_files = self._get_audio_files(self.root_dir, self.datasets)
        train_files, test_files = self._train_test_split_files(all_speaker_files, self.train_split)

        self._load_dataset('train', train_files)
        self._load_dataset('test', test_files)

        metadata = {
            'feature_shape': self.shape,
            'files_created': self.num_files_created,
            'speaker_id_map': self.speaker_id_map,
            'datasets': self.datasets,
            'train_test_split': [self.train_split, 1 - self.train_split],
            'num_records_per_file': self.buffer_flush_size
        }
        logger.info(f'Finished creating features, with metadata: {metadata}')
        with open(f'{self.target_dir}/metadata.json', 'w') as stream:
            json.dump(metadata, stream)
        return metadata

class GE2EBatchLoader(BatchLoader):

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
        speakers_per_batch,
        utterances_per_speaker,
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
        self.tf_serializer = GE2ESpectrogramSerializer()
        self.sr = sr
        self.num_files_created = {}
        self.speaker_id_map = {}

        self.shape = None
        self.train_split = train_split
        self.feature_buffer = deque()
        self.buffer_flush_size = tfrecord_examples_per_file
        self.speakers_per_batch = speakers_per_batch # N
        self.utterances_per_speaker = utterances_per_speaker # M

    def _build_batch(self, speaker_files, n_speakers, utterances_per_speaker):
        """
        Build the batches of N speakers each having M utterances and write them each 
        to a TFRecord file.
        """
        # each batch is [N*M, spectrogram_height, spectrogram_width]
        N, M = n_speakers, utterances_per_speaker
        batch = np.zeros((N*M, self.shape[0], self.shape[1]))
        labels = []

        speaker_set = random.sample(list(speaker_files.keys()), k=N)
        # TODO: add files & datasets to serialization
        for i, speaker in enumerate(speaker_set):
            j = i*M
            
            while j < (i+1)*M:
                f = speaker_files[speaker].pop()
                y, _ = librosa.load(f, sr=self.sr)
                features = list(self.extractor.as_features(y))

                while len(features) < M:
                    f = speaker_files[speaker].pop()
                    y, _ = librosa.load(f, sr=self.sr)
                    features += list(self.extractor.as_features(y))

                for feature in random.sample(features, k=M):
                    self._validate_numeric(feature)
                    batch[j,:,:] = feature
                    labels.append(speaker)
                    j += 1

        protobuf = self.tf_serializer.serialize(batch, labels)
        return protobuf

    def _write_batch(self, batch, subdir):
        subdir_path = f'{self.target_dir}/{subdir}'
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        path = f'{self.target_dir}/{subdir}/{subdir}_batch_{self.num_files_created.get(subdir, 0)+1:05d}.tfrecords'
        logger.info(f'Flushing batch into: {path}')
        with tf.io.TFRecordWriter(path) as writer:
            # TODO: we can collate batches now that we don't rely on tf.Dataset to do it
            writer.write(batch.SerializeToString())
        self.num_files_created[subdir] = self.num_files_created.get(subdir, 0) + 1

    def _update_shape(self, speaker_files):
        if self.shape is None:
            sample_speaker = list(speaker_files.keys())[0]
            f = speaker_files[sample_speaker][0]
            y, _ = librosa.load(f, sr=self.sr)
            feature = next(self.extractor.as_features(y))
            self.shape = feature.shape

    def _load_dataset(self, subdir, speaker_files):
        batches_remain = True
        while batches_remain:
            self._update_shape(speaker_files)
            batch = self._build_batch(speaker_files, self.speakers_per_batch, self.utterances_per_speaker)
            self._write_batch(batch, subdir)

            # remove speaker keys with files remaining < file threshold
            keys_to_remove = []
            for k, v in speaker_files.items():
                if len(v) < FILE_THRESHOLD:
                    keys_to_remove.append(k)
            for k in keys_to_remove:
                speaker_files.pop(k, None)

            # check if we have at leats N speakers left to build a batch
            batches_remain = len(speaker_files) >= self.speakers_per_batch

    def load(self):

        all_speaker_files = self._get_audio_files(self.root_dir, self.datasets)
        train_files, test_files = self._train_test_split_files(all_speaker_files, self.train_split)

        self._load_dataset('train', train_files)
        self._load_dataset('test', test_files)

        metadata = {
            'feature_shape': self.shape,
            'files_created': self.num_files_created,
            'batch_size': self.speakers_per_batch * self.utterances_per_speaker,
            'speakers_per_batch': self.speakers_per_batch,
            'utterances_per_speaker': self.utterances_per_speaker,
            'datasets': self.datasets
        }
        logger.info(f'Finished creating features, with metadata: {metadata}')
        with open(f'{self.target_dir}/metadata.json', 'w') as stream:
            json.dump(metadata, stream)
        return metadata