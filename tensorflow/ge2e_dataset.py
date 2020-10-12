import json
import os
import tensorflow as tf
from logger import logger
from serializer import SpectrogramSerializer

class GE2EDatasetLoader:

    def __init__(
        self,
        root_dir,
        batch_size
    ):
        self.train_dir = f'{root_dir}/train'
        self.test_dir = f'{root_dir}/test'
        self.batch_size = batch_size
        with open(f'{root_dir}/metadata.json', 'r') as stream:
            self.metadata = json.load(stream)
            self.example_dim = len(self.metadata['feature_shape'])
        self.serializer = SpectrogramSerializer()
        logger.info(f'Creating dataset with following metadata: {self.metadata}')

    def _prep_dataset(self, dir):
        tfrecord_file_names = list(filter(lambda f: f.endswith('.tfrecords'), os.listdir(dir)))
        tfrecord_file_paths = [ f'{dir}/{fname}' for fname in tfrecord_file_names ]
        logger.info(f'Dataset has {len(tfrecord_file_names)} files')
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
        d = raw_dataset.shuffle(len(tfrecord_file_names)).map(self.serializer.deserialize).batch(self.batch_size)
        return d

    def get_metadata(self):
        return self.metadata

    def get_single_train_batch(self):
        train_dataset = self._prep_dataset(self.train_dir)
        return next(iter(train_dataset))

    def get_single_test_batch(self):
        test_dataset = self._prep_dataset(self.test_dir)
        return next(iter(test_dataset))

    def get_datasets(self):
        train_dataset = self._prep_dataset(self.train_dir)
        test_dataset = self._prep_dataset(self.test_dir)
        return train_dataset, test_dataset

    def get_train_dataset(self):
        train_dataset = self._prep_dataset(self.train_dir)
        return train_dataset

    def get_test_dataset(self):
        test_dataset = self._prep_dataset(self.test_dir)
        return test_dataset