"""
The tf.keras.Model object formed from our config.yml representing our model
architecture.
"""
import tensorflow as tf

class SimilarityMatrixLayer(tf.keras.layers.Layer):

    def __init__(self, N, M, P):
        super(SimilarityMatrixLayer, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.random_uniform_initializer(-1.0, 1.0),
            shape=[1],
            trainable=True,
            name='similarity_matrix/weights'
        )
        self.b = tf.Variable(
            initial_value=tf.random_uniform_initializer(-2.0, 2.0),
            shape=[1],
            trainable=True,
            name='similarity_matrix/bias'
        )
        self.N = N # num unique speakers
        self.M = M # num utterances / speaker
        self.P = P # embedding dimension length

    def call(self, inputs):
        """
        Args:
            inputs: [NM x P]
        Returns:
            [NM x N]
        """
        # compute similarity matrix from the embedding layer
        N, M, P = self.N, self.M, self.P
        embedding_split = tf.reshape(inputs, shape=[N, M, P])
        center = None

        if center is None:
            # Eq (1); [N x P]
            # The centroid of tuple (e k1 , . . . , e kM ) represents the voiceprint built from M utterances
            # it is the average embedding vector for a given speaker, k
            centroids = tf.math.l2_normalize(tf.math.reduce_mean(embedding_split, axis=1), epsilon=1e-6)
            # Eq (8); [NM x P]
            centroids_except = tf.math.l2_normalize(
                tf.reshape(tf.reduce_sum(embedding_split, axis=1, keepdims=True) - embedding_split, shape=[N*M, P])
            )

            # Eq (9); [NM x N]
            S = tf.concat([
                tf.concat([
                    tf.reduce_sum(centroids[i:(i+1),:]*embedding_split[j,:,:], axis=1, keepdims=True) if i==j
                    else tf.reduce_sum(centroids[i:(i+1),:]*embedding_split[j,:,:], axis=1, keepdims=True)
                    for i in range(N)
                ], axis=1) 
            for j in range(N)], axis=0)
        # TODO: incorporate enrollments
        else :
            # If center(enrollment) exist, use it.
            S = tf.concat(
                [tf.concat([tf.math.reduce_sum(center[i:(i + 1), :] * embedded_split[j, :, :], axis=1, keepdims=True) for i
                            in range(N)],
                        axis=1) for j in range(N)], axis=0)
        # output shape [NM x N]; or num utterances by num speaker centroids
        return tf.clip_by_value(tf.abs(self.w)*S+self.b, clip_value_min=0.0, clip_value_max=1.0)

class SpeakerVerificationModel(tf.keras.Model):

    def __init__(self, conf, dataset_metadata, N, M):
        super(SpeakerVerificationModel, self).__init__()
        self.N = N
        self.M = M 
        self.layer_list = [
            tf.keras.layers.Input(
                shape=dataset_metadata['feature_shape'],
                batch_size=dataset_metadata['batch_size']
            )
        ]
        self.speakers_per_batch = dataset_metadata['speakers_per_batch']
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
            elif layer_type == 'similarity_matrix':
                self.layer_list += [SimilarityMatrixLayer(self.N, self.M, self.P)]
        return tf.keras.Sequential(self.layer_list)

    def call(self, inputs):
        return self.model(inputs)
