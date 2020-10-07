import json
import os
import tensorflow as tf
from logger import logger
from serializer import RawWaveformSerializer, SpectrogramSerializer

# TODO: batch_size is optional given GE2E
class DatasetLoader:

    def __init__(
        self,
        root_dir,
        batch_size,
        example_dim,
        feature_type
    ):
        self.train_dir = f'{root_dir}/train'
        self.test_dir = f'{root_dir}/test'
        self.batch_size = batch_size
        assert(feature_type in ['melspectrogram', 'raw']), 'Only feature_type raw or melspectrogram supported'
        if feature_type == 'melspectrogram':
            self.serializer = SpectrogramSerializer(example_dim=example_dim)
        else:
            self.serializer = RawWaveformSerializer(example_dim=example_dim)
        with open(f'{root_dir}/metadata.json', 'r') as stream:
            self.metadata = json.load(stream)
            self.metadata['batch_size'] = self.batch_size
        logger.info(f'Creating dataset with following metadata: {self.metadata}')

    def _prep_dataset(self, dir):
        tfrecord_file_names = list(filter(lambda f: f.endswith('.tfrecords'), os.listdir(dir)))
        tfrecord_file_paths = [ f'{dir}/{fname}' for fname in tfrecord_file_names ]
        logger.info(f'Dataset has {len(tfrecord_file_names)} files')
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
        batch_dataset = raw_dataset\
                            .shuffle(len(tfrecord_file_paths))\
                            .map(self.serializer.deserialize)\
                            .shuffle(self.batch_size * self.metadata['examples_per_file'])\
                            .batch(self.batch_size, drop_remainder=True)
        return batch_dataset

    def get_metadata(self):
        return self.metadata

    def get_single_train_batch(self):
        train_dataset = self._prep_dataset(self.train_dir)
        return next(iter(train_dataset))

    def get_single_test_batch(self):
        test_dataset = self._prep_dataset(self.test_dir)
        return next(iter(test_dataset))

    def get_dataset(self):
        train_dataset = self._prep_dataset(self.train_dir)
        test_dataset = self._prep_dataset(self.test_dir)
        return train_dataset, test_dataset
