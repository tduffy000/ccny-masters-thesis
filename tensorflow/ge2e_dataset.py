import json
import os
import tensorflow as tf
from logger import logger
from serializer import SpectrogramSerializer

class GE2EDatasetLoader:

    def __init__(
        self,
        root_dir
    ):
        self.train_dir = f'{root_dir}/train'
        self.test_dir = f'{root_dir}/test'
        with open(f'{root_dir}/metadata.json', 'r') as stream:
            self.metadata = json.load(stream)
            self.batch_size = self.metadata['batch_size']
            self.example_dim = len(self.metadata['feature_shape'])
        self.serializer = SpectrogramSerializer(example_dim=self.example_dim)
        logger.info(f'Creating dataset with following metadata: {self.metadata}')

    def _prep_dataset(self, dir):
        tfrecord_file_names = list(filter(lambda f: f.endswith('.tfrecords'), os.listdir(dir)))
        tfrecord_file_paths = [ f'{dir}/{fname}' for fname in tfrecord_file_names ]
        logger.info(f'Dataset has {len(tfrecord_file_names)} files')
        raw_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
        # TODO: need to figure out how to shuffle the batches
        batch_dataset = raw_dataset\
                            .map(self.serializer.deserialize)\
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
        # test_dataset = self._prep_dataset(self.test_dir)
        return train_dataset#, test_dataset
