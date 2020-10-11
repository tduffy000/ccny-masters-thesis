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
        train_datasets,
        test_datasets,
        target_dir,
        window_length, # in seconds
        overlap_percent, # in percent
        frame_length, # in seconds
        hop_length, # in seconds
        n_fft=512,
        n_mels=40,
        sr=16000,
        trim_top_db=30,
        speakers_per_batch=20,
        utterances_per_speaker=10
    ):
        self.root_dir = root_dir
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
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
        self.num_files_created = {'train': 0, 'test': 0}
        
        self.shape = None

        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker

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

    def _get_audio_files(self, speaker_files, source_dir):
        """
        Walks the source directory tree and pulls out .flac files, creating a
        mapping of speaker_id -> [speaker specific files].

        :param source_dir: The root directory containing audio files to walk.
        :type source_dir: str
        :return: {speaker_id -> [speaker specific files]}
        """
        for root, dirs, files in os.walk(source_dir):
            if files and not dirs:
                # how do you pull out a VoxCeleb ID?
                speaker_id = int(root.split("/")[-2]) # this is specific to LibriSpeech
                d = speaker_files.get(speaker_id, [])
                paths = [ f'{root}/{f}' for f in filter(self._is_audio_file, files) ]
                speaker_files[speaker_id] = d + paths
        return speaker_files

    def _update_shape(self, speaker_files):
        if self.shape is None:
            sample_speaker = list(speaker_files.keys())[0]
            f = speaker_files[sample_speaker][0]
            y, _ = librosa.load(f, sr=self.sr)
            feature = next(self.extractor.as_features(y))
            self.shape = feature.shape

    def _build_batch(self, speaker_files, n_speakers, utterances_per_speaker):
        """
        Build the batches of N speakers each having M utterances and write them each 
        to a TFRecord file.
        """
        # each batch is [N*M, spectrogram_height, spectrogram_width]
        N = n_speakers
        M = utterances_per_speaker
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

                # if we don't get M features back
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

    def _write_batch(self, batch, is_train=True):
        subdir = 'train' if is_train else 'test'
        subdir_path = f'{self.target_dir}/{subdir}'
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        path = f'{self.target_dir}/{subdir}/{subdir}_batch_{self.num_files_created.get(subdir)+1:05d}.tfrecords'
        logger.info(f'Flushing batch into: {path}')
        with tf.io.TFRecordWriter(path) as writer:
            # TODO: we can collate batches now that we don't rely on tf.Dataset to do it
            writer.write(batch.SerializeToString())
        self.num_files_created[subdir] = self.num_files_created.get(subdir, 0) + 1

    def _load_dataset(self, datasets, is_train):
        speaker_files = {}
        for dataset, subsets in datasets.items():
            dataset_path = f'{self.root_dir}/{dataset}'
            for subset in subsets:
                subset_path = f'{dataset_path}/{subset}'
                # how to concatenate multiple datasets here?
                speaker_files = self._get_audio_files(speaker_files, subset_path)
        batches_remain = True
        while batches_remain:
            self._update_shape(speaker_files)
            batch = self._build_batch(speaker_files, self.speakers_per_batch, self.utterances_per_speaker)
            self._write_batch(batch, is_train)

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

        self._load_dataset(self.train_datasets, is_train=True)
        self._load_dataset(self.test_datasets, is_train=False)

        metadata = {
            'feature_shape': self.shape,
            'files_created': self.num_files_created,
            'batch_size': self.speakers_per_batch * self.utterances_per_speaker,
            'speakers_per_batch': self.speakers_per_batch,
            'utterances_per_speaker': self.utterances_per_speaker,
            'datasets': {
                'train': self.train_datasets,
                'test': self.test_datasets
            }
        }
        logger.info(f'Finished creating features, with metadata: {metadata}')
        with open(f'{self.target_dir}/metadata.json', 'w') as stream:
            json.dump(metadata, stream)
        return metadata