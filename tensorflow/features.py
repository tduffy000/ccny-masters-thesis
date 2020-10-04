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
        trim_top_db=30
    ):
        self.sr = sr

        # initial windowing
        self.window_length = int(window_length * sr)
        self.frame_length = int(frame_length * sr)
        self.hop_length = int(hop_length * sr)
        self.overlap = int(self.window_length * overlap_percent)
        self.trim_top_db = trim_top_db

        # conversion of raw waveform windows into mel spectrogram
        self.n_fft=n_fft
        self.n_mels=n_mels
        self.log_lift = 1e-6

    def _get_intervals(self, y):
        """
        Converts a single channel raw waveform into multiple windows (rows in the target)
        where each window is of length `window_length` and they have an overlap of `overlap`.

        :param y: The single channel raw waveform loaded from librosa.load().
        :type y: numpy.ndarray shape=[num_samples, ]
        :return: The windows of this waveform.
        :rtype: numpy.ndarray shape=[num_windows, window_length]
        """
        # first we split the raw waveform into utterances using librosa's voice activity detection
        intervals = [ 
            itv for itv in librosa.effects.split(y, top_db=self.trim_top_db) if itv[1] - itv[0] >= self.window_length
        ]
        # then for each utterance, break them apart into windows
        windows = []
        for (left, right) in intervals:
            utterance = y[left:right]
            frames = librosa.util.frame(utterance, frame_length=self.window_length, hop_length=self.overlap, axis=0)
            for frame in frames:
                windows.append(frame)
        return windows

    def as_melspectrogram(self, y):
        """
        :param y: The raw waveform.
        :type y: numpy.ndarray shape=[samples,]
        :return: A log-spaced melspectrogram.
        :rtype: numpy.ndarray
        """
        for w in self._get_intervals(y):
            feature = librosa.feature.melspectrogram(
                w,
                sr=self.sr,
                n_fft=self.n_fft,
                n_mels=self.n_mels,
                win_length=self.frame_length,
                hop_length=self.hop_length
            )
            yield np.log10(feature + self.log_lift)
