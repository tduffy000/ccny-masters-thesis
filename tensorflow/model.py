"""
The tf.keras.Model object formed from our config.yml representing our model
architecture.
"""
import tensorflow as tf

class SpeakerSimilarityMatrixLayer(tf.keras.layers.Layer):

    def __init__(self, n_speakers, utterances_per_speaker, embedding_length):
        super(SpeakerSimilarityMatrixLayer, self).__init__()
        self.W = tf.Variable(name='W', trainable=True, initial_value=10.)
        self.b = tf.Variable(name='b', trainable=True, initial_value=-5.)
        self.N = n_speakers
        self.M = utterances_per_speaker
        self.P = embedding_length

    def call(self, inputs):
        """
        Args:
            inputs: output from the final Dense(self.P) embedding layer, representing each
                    speakers "voiceprint" for a given utterance.
        Returns:
            An [NM x N] cosine similarity matrix comparing the NM utterances in each column
            to the N centroids (representing the averaged embedding for a given speaker).
        """
        # [n_speakers x utterances x embedding_length]
        inputs = tf.math.l2_normalize(inputs, axis=1)
        utterance_embeddings = tf.reshape(inputs, shape=[self.N, self.M, self.P])

        # the averaged embeddings for each speaker: [n_speakers x embedding_length]
        centroids = tf.math.l2_normalize(
            tf.reduce_mean(utterance_embeddings, axis=1),
            axis=1
        )
        # now we need every utterance_embedding's cosine similarity with those centroids
        # returning: [n_speakers * utterances x n_speakers (or n_centroids)]
        S = tf.concat(
            [tf.matmul(utterance_embeddings[i], centroids, transpose_b=True) for i in range(self.N)],
            axis=0
        )
        return tf.abs(self.W) * S + self.b


class SpeakerVerificationModel(tf.keras.Model):

    def __init__(self, conf, dataset_metadata):
        super(SpeakerVerificationModel, self).__init__()
        self.layer_list = [
            tf.keras.layers.Input(
                shape=dataset_metadata['feature_shape']
            )
        ]
        self.n_speakers = dataset_metadata['speakers_per_batch']
        self.utterances_per_speaker = dataset_metadata['utterances_per_speaker']
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
    def get_lstm(units, return_sequences, activation='tanh'):
        return [
            tf.keras.layers.LSTM(units=units, return_sequences=return_sequences, activation=activation)
        ]

    @staticmethod
    def get_lstm_cells():
        pass

    @staticmethod
    def get_gru_cells():
        pass

    @staticmethod
    def get_gru(units, return_sequences, activation='tanh'):
        return [
            tf.keras.layer.GRU(units=units, return_sequences=return_sequences, activation=activation)
        ]

    @staticmethod
    def get_bidirectional(inner, units, return_sequences=False):
        if inner == 'lstm':
            inner_layer = tf.keras.layers.LSTM(units=units, return_sequences=return_sequences)
        elif inner == 'gru':
            inner_layer = tf.keras.layers.GRU(units=units, return_sequences=return_sequences)
        return [tf.keras.layers.Bidirectional(inner_layer)]

    @staticmethod
    def get_global_pooling(layer_type):
        if layer_type == 'average':
            return [tf.keras.layers.GlobalAveragePooling1D()]
        elif layer_type == 'max':
            return [tf.keras.layers.GlobalMaxPooling1D()]

    @staticmethod
    def get_local_pooling(layer_type):
        if layer_type == 'average':
            return [tf.keras.layers.AveragePooling1D()]
        elif layer_type == 'max':
            return [tf.keras.layers.MaxPooling1D()]

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
                self.layer_list += self.get_lstm(units=layer['units'], return_sequences=layer.get('return_sequences', False))
            elif layer_type == 'gru':
                self.layer_list += self.get_gru(units=layer['units'], return_sequences=layer.get('return_sequences', False))
            elif layer_type == 'bidirectional':
                self.layer_list += self.get_bidirectional(layer['inner'], layer['units'], layer.get('return_sequences', False))
            elif layer_type == 'flatten':
                self.layer_list += [tf.keras.layers.Flatten()]
            elif layer_type == 'fc':
                self.layer_list += self.get_fc(layer['nodes'])
            elif layer_type == 'embedding':
                self.P = layer['nodes']
                self.layer_list += self.get_fc(layer['nodes'])
            elif layer_type == 'softmax':
                self.layer_list += self.get_fc(self.n_speakers, activation='softmax')
            elif layer_type == 'similarity_matrix':
                self.layer_list += [SpeakerSimilarityMatrixLayer(self.n_speakers, self.utterances_per_speaker, layer['embedding_length'])]
        return tf.keras.Sequential(self.layer_list)

    def call(self, inputs):
        return self.model(inputs)
