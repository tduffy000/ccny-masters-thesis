"""
The tf.keras.Model object formed from our config.yml representing our model
architecture.
"""
import tensorflow as tf

# https://arxiv.org/pdf/1803.05427.pdf
# https://towardsdatascience.com/tensorflow-speech-recognition-challenge-solution-outline-9c42dbd219c9
# https://towardsdatascience.com/debugging-a-machine-learning-model-written-in-tensorflow-and-keras-f514008ce736
# https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
class SpeakerVerificationModel(tf.keras.Model):

    # input shape from dataset metadata
    def __init__(self, conf, dataset_metadata):
        super(SpeakerVerificationModel, self).__init__()
        self.n_classes = len(dataset_metadata['speaker_id_mapping'])
        self.layer_list = [
            tf.keras.layers.Input(shape=dataset_metadata['shape'], batch_size=dataset_metadata['batch_size'])
        ]
        self.model = self._parse_layer_conf(conf['layers'])

    @staticmethod
    def get_conv1d(
        filters,
        kernel_size,
        dropout_prob=0.1,
        use_batchnorm=True,
        pooling='max'
    ):
        return [
            tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(dropout_prob)
        ]

    @staticmethod
    def get_fc(nodes, activation='relu'):
        return [tf.keras.layers.Dense(nodes, activation=activation)]

    def _parse_layer_conf(self, conf):
        for layer_name, layer_conf in conf.items():
            if layer_name == 'conv1d':
                for _ in range(layer_conf['n']):
                    self.layer_list += self.get_conv1d(layer_conf['filters'], layer_conf['kernel_size'])
            elif layer_name == 'fc':
                self.layer_list += [tf.keras.layers.Flatten()]
                for _ in range(layer_conf['n']):
                    self.layer_list += self.get_fc(layer_conf['nodes'])
            elif layer_name == 'embedding':
                self.layer_list += self.get_fc(layer_conf['nodes'])
            elif layer_name == 'output':
                self.layer_list += self.get_fc(self.n_classes, 'softmax')
        return tf.keras.Sequential(self.layer_list)

    def call(self, inputs):
        return self.model(inputs)
