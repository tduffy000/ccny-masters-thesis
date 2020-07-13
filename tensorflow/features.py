import librosa
import numpy as np

# TODO: some add the 1st and 2nd derivatives by way of librosa.feature.delta
class FeatureExtractor:

    def __init__(
        self,
        window_length, # in seconds
        overlap_percent, # in percent
        frame_length, # in seconds
        hop_length, # in seconds
        n_fft=512,
        n_mels=40,
        sr=16000,
        use_preemphasis=True,
        preemphasis_coef=0.97,
        trim_silence=True,
        trim_top_db=30,
        normalization=None,
        window='hamming' # unused at the moment
    ):
        self.sr = sr

        # initial windowing
        self.window_length = int(window_length * sr)
        self.overlap = int(self.window_length * overlap_percent)
        self.trim_silence = trim_silence
        self.trim_top_db = trim_top_db
        self.use_preemphasis = use_preemphasis
        self.preemphasis_coef = preemphasis_coef

        # conversion of raw waveform windows into mel spectrogram
        self.max_normalization, self.mean_normalization = False, False
        if normalization == 'mean':
            self.mean_normalization = True
        if normalization == 'max':
            self.max_normalization = True
        self.frame_length = int(frame_length * sr)
        self.hop_length = int(hop_length * sr)
        self.n_fft=n_fft
        self.n_mels=n_mels
        self.log_lift = 1e-6 # required to avoid introducing np.nan

    def _get_windows(self, v):
        """
        Converts a single channel raw waveform into multiple windows (rows in the target)
        where each window is of length `window_length` and they have an overlap of `overlap`.

        :param v: The single channel raw waveform.
        :type v: numpy.ndarray shape=[num_samples, ]
        :return: The windows of this waveform.
        :rtype: numpy.ndarray shape=[num_windows, window_length]
        """
        if self.trim_silence:
            v, _ = librosa.effects.trim(v, top_db=self.trim_top_db)
        if self.use_preemphasis:
            v = librosa.effects.preemphasis(v, coef=self.preemphasis_coef)
        if self.mean_normalization:
            v -= np.mean(v)
        if self.max_normalization:
            v = v / np.max(v)
        if v.shape[0] < self.window_length:
            return []
        return librosa.util.frame(v, frame_length=self.window_length, hop_length=self.overlap, axis=0)

    def as_melspectrogram(self, v):
        """
        :param v: The raw waveform.
        :type v: numpy.ndarray shape=[samples,]
        :return: A melspectrogram.
        :rtype: numpy.ndarray
        """
        for w in self._get_windows(v):
            feature = librosa.feature.melspectrogram(
                w,
                sr=self.sr,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                win_length=self.frame_length,
                hop_length=self.hop_length
            )
            yield np.log(feature + self.log_lift)
