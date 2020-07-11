"""
The tf.data.Dataset sourced from our TFRecord files written by the transformer.
"""
# https://github.com/jkjung-avt/keras_imagenet/tree/ece09ff909cf7db3640f849a7bf7036330574d29
import json
import os
import tensorflow as tf
from logger import logger
from serializer import SpectrogramSerializer

class DatasetLoader:

    def __init__(
        self,
        source_dir,
        url,
        batch_size,
        example_dim
    ):
        self.source_dir = f'{source_dir}/{url}'
        self.batch_size = batch_size
        self.serializer = SpectrogramSerializer(example_dim=example_dim)
        with open(f'{self.source_dir}/metadata.json', 'r') as stream:
            self.metadata = json.load(stream)
        logger.info(f'Creating dataset with following metadata: {self.metadata}')

    def load_dataset(self):
        tfrecord_file_names = list(filter(lambda f: f.endswith('.tfrecords'), os.listdir(self.source_dir)))
        tfrecord_file_paths = [ f'{self.source_dir}/{fname}' for fname in tfrecord_file_names ]
        logger.info(f'Dataset has {len(tfrecord_file_names)} files')
        self.raw_dataset = tf.data.TFRecordDataset(tfrecord_file_paths)
        self.batch_dataset = self.raw_dataset\
                                 .shuffle(len(tfrecord_file_paths))\
                                 .map(self.serializer.deserialize)\
                                 .shuffle(self.batch_size * self.metadata['examples_per_file'])\
                                 .batch(self.batch_size)
        return self.batch_dataset
