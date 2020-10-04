"""
The tf.keras.Model object formed from our config.yml representing our model
architecture.
"""
import tensorflow as tf

# https://arxiv.org/pdf/1803.05427.pdf
# https://towardsdatascience.com/tensorflow-speech-recognition-challenge-solution-outline-9c42dbd219c9
# https://towardsdatascience.com/debugging-a-machine-learning-model-written-in-tensorflow-and-keras-f514008ce736
# https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33

# https://github.com/Janghyun1230/Speaker_Verification
# https://www.tensorflow.org/guide/keras/train_and_evaluate#handling_losses_and_metrics_that_dont_fit_the_standard_signature
class SpeakerVerificationModel(tf.keras.Model):

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
        layers = [tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)]
        if use_batchnorm:
            layers += [tf.keras.layers.BatchNormalization()]
        if pooling == 'max':
            layers += []
        if pooling == 'avg':
            layers += []
        if dropout_prob > 0.0:
            layers += [tf.keras.layers.Dropout(dropout_prob)]
        return layers

    @staticmethod
    def get_conv2d(
        filters,
        kernel_size,
        dropout_prob=0.1,
        use_batchnorm=True,
        pooling='max'
    ):
        layers = [tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size)]
        if use_batchnorm:
            layers += [tf.keras.layers.BatchNormalization()]
        if pooling == 'max':
            layers += []
        if pooling == 'avg':
            layers += []
        if dropout_prob > 0.0:
            layers += [tf.keras.layers.Dropout(dropout_prob)]
        return layers

    @staticmethod
    def get_lstm(units, activation='tanh'):
        return [tf.keras.layers.LSTM(units=units, activation=activation)]

    @staticmethod
    def get_gru(units, activation='tanh'):
        return [tf.keras.layer.GRU(units=units, activation=activation)]

    @staticmethod
    def get_bidirectional(layer_type, **kwargs):
        if layer_type == 'lstm':
            inner_layer = tf.keras.layers.LSTM(kwargs['units'])
        elif layer_type == 'gru':
            inner_layer = tf.keras.layers.GRU(kwargs['units'])
        return [tf.keras.layers.Bidirectional(inner_layer)]

    @staticmethod
    def get_global_pooling():
        pass

    @staticmethod
    def get_fc(nodes, activation='relu'):
        return [tf.keras.layers.Dense(nodes, activation=activation)]

    def _parse_layer_conf(self, conf):
        for layer_conf in conf:
            if isinstance(layer_conf, dict):
                # for layers requiring parameters
                layer_type = list(layer_conf.keys())[0]
                layer = layer_conf[layer_type]
            elif isinstance(layer_conf, str):
                # for things like 'output' or 'flatten' with no parameters
                layer_type = layer_conf
            if layer_type == 'conv1d':
                self.layer_list += self.get_conv1d(
                    layer['filters'],
                    layer['kernel_size']
                )
            elif layer_type == 'lstm':
                self.layer_list += self.get_lstm(units=layer['units'])
            elif layer_type == 'gru':
                self.layer_list += self.get_gru(units=layer['units'])
            elif layer_type == 'flatten':
                self.layer_list += [tf.keras.layers.Flatten()]
            elif layer_type == 'fc':
                self.layer_list += self.get_fc(layer['nodes'])
            elif layer_type == 'embedding':
                self.layer_list += self.get_fc(layer['nodes'])
            elif layer_type == 'output':
                self.layer_list += self.get_fc(self.n_classes, 'softmax')
        return tf.keras.Sequential(self.layer_list)

    def call(self, inputs):
        return self.model(inputs)
