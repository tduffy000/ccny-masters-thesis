"""
The tf.keras.Model object formed from our config.yml representing our model
architecture.
"""
import tensorflow as tf

class SpeakerVerificationModel(tf.keras.Model):

    def __init__(self, conf, dataset_metadata):
        super(SpeakerVerificationModel, self).__init__()
        self.layer_list = [
            tf.keras.layers.Input(
                shape=dataset_metadata['feature_shape']
            )
        ]
        self.n_speakers = len(dataset_metadata['speaker_id_map'])
        self.model = self._parse_layer_conf(conf['layers'])

    @staticmethod
    def get_conv1d(
        filters,
        kernel_size,
        dropout_prob=0.1,
        use_batchnorm=True,
        pooling=None,
        pool_size=None
    ):
        layers = [tf.keras.layers.Conv1D(filters=filters, kernel_size=kernel_size)]
        if use_batchnorm:
            layers += [tf.keras.layers.BatchNormalization()]
        if pooling == 'max':
            layers += [tf.keras.layers.MaxPooling1D(pool_size=pool_size)]
        if pooling == 'avg':
            layers += [tf.keras.layers.AveragePooling1D(pool_size=pool_size)]
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

    # @staticmethod
    # def get_global_pooling():
    #     if layer_type == ''

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
                self.P = layer['nodes']
                self.layer_list += self.get_fc(layer['nodes'])
            elif layer_type == 'softmax':
                self.layer_list += self.get_fc(self.n_speakers, activation='softmax')
        return tf.keras.Sequential(self.layer_list)

    def call(self, inputs):
        return self.model(inputs)
