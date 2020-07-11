"""
The tf.data.Dataset sourced from our TFRecord files written by the transformer.
"""
# https://github.com/jkjung-avt/keras_imagenet/tree/ece09ff909cf7db3640f849a7bf7036330574d29
import tensorflow as tf
from serializer import SpectrogramSerializer

class DatasetLoader:

    def __init__(
        self,
        source_dir,
        batch_size,
        example_dim
    ):
        self.source_dir = source_dir
        self.batch_size = batch_size
        self.serializer = SpectrogramSerializer(example_dim=example_dim)
        with open(f'{source_dir}/metadata.json', 'r') as stream:
            self.metadata = json.load(stream)

    def load_dataset(self):
        tfrecords_file = list(filter(lambda f: f.endswith('.tfrecords'), os.listdir(self.source_dir)))
        self.raw_dataset = tf.data.TFRecordDataset(tfrecords_file)
        self.batch_dataset = self.dataset\
                                 .shuffle(len(tfrecords_file))\
                                 .map(self.serializer.deserialize)\
                                 .shuffle(self.batch_size * self.metadata['examples_per_file'])\
                                 .batch(self.batch_size)
        return self.batch_dataset
