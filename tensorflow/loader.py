import json
import random
import os
import shutil

import numpy as np
import tensorflow as tf

from logger import logger

class BatchLoader:

    def __init__(
        self,
        root_dir,
        datasets,
        target_dir,
        feature_extractor,
        serializer,
        speakers_per_batch,
        utterances_per_speaker
    ):
        self.root_dir = root_dir
        self.datasets = datasets
        self.target_dir = target_dir

        self.feature_extractor = feature_extractor
        self.feature_shape = None
        self.serializer = serializer

        self.speakers_per_batch = speakers_per_batch
        self.utterances_per_speaker = utterances_per_speaker

        self.num_files_created = {}
        self.speaker_id_mapping = {}

    @staticmethod
    def is_audio_file(fname):
        for ftype in ['.flac', '.wav', '.mp3']:
            if fname.endswith(ftype):
                return True
        return False

    def update_feature_shape(self, path):
        if self.feature_shape is None:
            self.feature_shape = self.feature_extractor.get_feature_shape(path)

    def update_speaker_id_mapping(self, speaker_file_mapping):
        for speaker_id in speaker_file_mapping:
            if speaker_id not in self.speaker_id_mapping:
                self.speaker_id_mapping[speaker_id] = len(self.speaker_id_mapping)

    def get_files(self, datasets):
        # mapping of speaker_id -> [(n_features, path), ...]
        file_mapping = {}
        for parent, subsets in datasets.items():
            for subset in subsets:
                subset_path = os.path.join(self.root_dir, parent, subset)
                for root, dirs, files in os.walk(subset_path):
                    if files and not dirs:
                        speaker_id = f'{parent}/{subset}/{root.split("/")[-2]}' # both LibriSpeech and VoxCeleb1 have it as this level

                        audio_files = list(filter(self.is_audio_file, files))
                        for f in audio_files:
                            abs_path = os.path.join(root, f)
                            num_windows, windows = self.feature_extractor.get_intervals(abs_path)
                            speaker_files = file_mapping.get(speaker_id, [])
                            if num_windows > 0:
                                self.update_feature_shape(abs_path)
                                file_mapping[speaker_id] = speaker_files + [(num_windows, abs_path)]
        return file_mapping

    def build_batch(self, file_mapping, n_speakers, utterances_per_speaker):
        N, M = n_speakers, utterances_per_speaker
        batch = np.zeros((N*M, self.feature_shape[0], self.feature_shape[1]))
        speakers, labels = [], []

        batch_speakers = random.sample(list(file_mapping.keys()), k=N)

        # for every speaker i = 0, ..., N; get M utterances for the batch
        for i, speaker in enumerate(batch_speakers):
            mapped_id = self.speaker_id_mapping[speaker]
            j = i*M
            while j < (i+1)*M:

                _, f = file_mapping[speaker].pop()
                features = list(self.feature_extractor.as_features(f))
                while len(features) < M:
                    _, f = file_mapping[speaker].pop()
                    features += list(self.feature_extractor.as_features(f))

                random.shuffle(features)
                for feature in features[:M]:
                    batch[j,:,:] = feature
                    speakers.append(speaker)
                    labels.append(mapped_id)
                    j += 1
                    
        pb = self.serializer.serialize(batch, speakers, labels)
        return pb

    # TODO: put this in the serializer?
    def write_batch(self, pb, subdir='train'):
        subdir_path = f'{self.target_dir}/{subdir}'
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)
        path = f'{self.target_dir}/{subdir}/{subdir}_batch_{self.num_files_created.get(subdir, 0)+1:05d}.tfrecords'
        logger.info(f'Flushing batch into: {path}')
        with tf.io.TFRecordWriter(path) as writer:
            # TODO: we can collate batches now that we don't rely on tf.Dataset to do it
            writer.write(pb.SerializeToString())
        self.num_files_created[subdir] = self.num_files_created.get(subdir, 0) + 1

    @staticmethod
    def filter_keys_for_batches(files, utterances_per_speaker):
        total_windows = {}
        for speaker, file_count_pairs in files.items():
            total_windows[speaker] = sum([n_windows for n_windows, _ in file_count_pairs])

        speakers_to_pop = [
            speaker for speaker, n_windows in total_windows.items() if n_windows < utterances_per_speaker 
        ]

        for speaker in speakers_to_pop:
            files.pop(speaker, None)

    def load(self):

        # load training data
        in_sample_datasets = self.datasets['train']
        in_sample_speaker_file_mapping = self.get_files(in_sample_datasets)
        self.update_speaker_id_mapping(in_sample_speaker_file_mapping)
        
        shutil.rmtree(f'{self.target_dir}/train', ignore_errors=True)
        while True: # TODO: DRY candidate
            self.filter_keys_for_batches(in_sample_speaker_file_mapping, self.utterances_per_speaker)
            if len(in_sample_speaker_file_mapping) < self.speakers_per_batch:
                break
            batch = self.build_batch(
                in_sample_speaker_file_mapping,
                self.speakers_per_batch,
                self.utterances_per_speaker
            )
            self.write_batch(batch, subdir='train')

        test_dataset_conf = self.datasets.get('test', None)
        if test_dataset_conf is not None:
            shutil.rmtree(f'{self.target_dir}/test', ignore_errors=True)
            if 'out_of_sample' in test_dataset_conf:
                test_datasets = test_dataset_conf['out_of_sample']
                out_of_sample_speaker_file_mapping = self.get_files(test_datasets)
                self.update_speaker_id_mapping(out_of_sample_speaker_file_mapping)
                while True: # TODO: DRY candidate
                    self.filter_keys_for_batches(
                        out_of_sample_speaker_file_mapping,
                        self.utterances_per_speaker
                    )
                    if len(out_of_sample_speaker_file_mapping) < self.speakers_per_batch:
                        break
                    batch = self.build_batch(
                        out_of_sample_speaker_file_mapping,
                        self.speakers_per_batch,
                        self.utterances_per_speaker
                    )
                    self.write_batch(batch, subdir='test')

        metadata = {
            'feature_shape': self.feature_shape,
            'files_created': self.num_files_created,
            'batch_size': self.speakers_per_batch * self.utterances_per_speaker,
            'speakers_per_batch': self.speakers_per_batch,
            'utterances_per_speaker': self.utterances_per_speaker,
            'datasets': self.datasets,
            'speaker_id_mapping': self.speaker_id_mapping
        }
        logger.info(f'Finished creating features, with metadata: {metadata}')
        with open(f'{self.target_dir}/metadata.json', 'w') as stream:
            json.dump(metadata, stream)
        return metadata