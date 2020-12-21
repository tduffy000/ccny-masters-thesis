## Tensorflow model
This directory contains code related to the feature engineering pipelines, model training, and model serialization / compression.

In [`main.py`](./main.py) we support the following flags for running different parts of the program in isolation, or together.:
* `--feature_engineering`: Perform the feature engineering steps to create a `tf.data.TFRecordDataset`
* `--train`: Train the model specified in the config.
* `--freeze_model`: Freeze the model using `tf.keras.models.save`.
* `--convert_to_lite`: Convert the model to a `tf.lite.micro` object.

The hyperparameters for each one of these steps in contained in a config in the [`conf`](./conf) directory. You will need to provide the config file path via the `--config_file` flag. 

### Feature Engineering
Our feature engineering pipeline consists of generating log-mel spectrograms utilizing the `librosa` library. For example, we define a `FeatureExtractor` (as defined in [features.py](./features.py)), 
```python
class SpectrogramExtractor(FeatureExtractor):

    def __init__(
        self,
        window_length,   # in seconds
        overlap_percent, # in percent
        frame_length,    # in seconds
        hop_length,      # in seconds
        sr=16000,
        n_fft=512,
        n_mels=40,
        trim_top_db=30,
        use_mean_normalization=False
    ):
        super().__init__(
            window_length=window_length,
            overlap_percent=overlap_percent,
            sr=sr,
            trim_top_db=trim_top_db
        )
        self.hop_length = int(hop_length * sr)
        self.frame_length = int(frame_length * sr)
        self.n_fft=n_fft
        self.n_mels=n_mels
        self.log_lift = 1e-6
        self.mean_normalize = use_mean_normalization

    def as_features(self, path):
        _, windows = self.get_intervals(path)
        for w in windows:
            feature = librosa.feature.melspectrogram(
                w,
                sr=self.sr,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                win_length=self.frame_length,
                hop_length=self.hop_length
            )
            self.validate_numeric(feature)
            S = np.log10(feature + self.log_lift)
            if self.mean_normalize:
                S -= np.mean(S, axis=0)
            yield S
```
which chunks an arbitrarily long audio file into different windows, via the `get_intervals(file_path)` method, and then generate a spectrogram for each constant time chunk.

Ultimately, those features are generated over a directory containing a number of files which have some speaker-specific identifier (which we'll need for batch construction). Those files and the spectrogram pipeline are represented by a `tf.data.Dataset` object, specifically a `tf.data.TFRecordDataset`. The conversion procress from raw files to a `tf.data.TFRecordDataset` is performed in [`loader.py`](./loader.py).

## Model
In training our model, we make use of a custom loss layer, the `SpeakerSimilarityMatrix` layer
```python
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
```
which is the final layer in our training process. We then calculcate the loss over this layer, as defined in [`loss.py`](./loss.py),
```python
def get_embedding_loss(N, M):
    """
    Get the loss function relating to the Generalized End-to-End model architecture which 
    returns a cosine similarity matrix of each speaker's embedding to the speaker centroids. 

    Args:
        N: The number of unique speakers in a batch.
        M: The number of utterances from each speaker in a batch.
    Returns:
        A closure loss function taking in the similarity matrix generated by model.SpeakerSimilarityMatrixLayer.
    """
    def loss(_, S):
        S_correct = tf.concat([S[i*M:(i+1)*M, i:(i+1)] for i in range(N)], axis=0)

        l = tf.math.reduce_sum(
            -S_correct + tf.math.log(tf.math.reduce_sum(tf.exp(S), axis=1, keepdims=True) + 1e-6)
        )
        return l
    return loss
```
Where the hope is by minimizing our loss, we push speaker utterance embeddings toward their respective centroids and away from the centroids of other speakers.