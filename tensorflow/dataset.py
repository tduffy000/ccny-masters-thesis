import json
import os
import tensorflow as tf
from logger import logger
from serializer import SpectrogramSerializer, GE2ESpectrogramSerializer

# TODO: we might not need the metadata here anymore
SHUFFLE_SIZE = 5000

class DatasetLoader:

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
        self.records_per_file = self.metadata['batch_size'] # TODO: address this when writing more than one batch to a file
        self.serializer = SpectrogramSerializer()

    # https://github.com/tensorflow/tensorflow/issues/14857
    def _prep_dataset(self, dir):
        tfrecord_file_names = list(filter(lambda f: f.endswith('.tfrecords'), os.listdir(dir)))
        tfrecord_file_paths = [ f'{dir}/{fname}' for fname in tfrecord_file_names ]
        dataset = tf.data.Dataset.from_tensor_slices(tfrecord_file_paths)
        num_shards = len(tfrecord_file_paths)
        dataset = dataset.shuffle(num_shards).interleave(lambda f: tf.data.TFRecordDataset(f), deterministic=False,cycle_length=num_shards).shuffle(self.records_per_file).map(self.serializer.deserialize)
        if self.batch_size is not None: # can be already provided
            dataset = dataset.batch(self.batch_size)
        return dataset

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

class GE2EDatasetLoader(DatasetLoader):

    def __init__(self, root_dir):
        super().__init__(root_dir, None)
        self.serializer = GE2ESpectrogramSerializer()