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
    def get_lstm():
        pass

    @staticmethod
    def get_global_pooling():
        pass

    @staticmethod
    def get_fc(nodes, activation='relu'):
        return [tf.keras.layers.Dense(nodes, activation=activation)]

    # this can use the ModelConfig object
    def _parse_layer_conf(self, conf):
        for layer_name, layer_conf in conf.items():
            if layer_name == 'conv1d':
                for layer in layer_conf:
                    self.layer_list += self.get_conv1d(
                        layer['filters'],
                        layer['kernel_size']
                    )
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
