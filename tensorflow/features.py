import librosa
import numpy as np

class FeatureExtractor:

    def __init__(
        self,
        window_length,   # in seconds
        overlap_percent, # in percent
        sr=16000,
        trim_top_db=30
    ):
        self.sr = sr

        # initial windowing
        self.window_length = int(window_length * sr)
        self.overlap = int(self.window_length * overlap_percent)
        self.trim_top_db = trim_top_db

    @staticmethod
    def validate_numeric(feature):
        nans_found = np.sum(np.isnan(feature))
        infs_found = np.sum(np.isinf(feature))
        if nans_found > 0 or infs_found > 0:
            logger.fatal('Found a NaN or Infinity!')
            raise ValueError('Features must have real-valued elements.')

    def get_feature_shape(self, path):
        try:
            return next(self.as_features(path)).shape
        except StopIteration:
            return None

    def get_intervals(self, path):
        """
        Converts a single channel raw waveform into multiple windows (rows in the target)
        where each window is of length `window_length` and they have an overlap of `overlap`.

        :param y: The single channel raw waveform loaded from librosa.load().
        :type y: numpy.ndarray shape=[num_samples, ]
        :return: Number of intervals and the windowed waveform.
        :rtype: (int, numpy.ndarray shape=[num_windows, window_length])
        """
        y, _ = librosa.load(path, self.sr)
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
        return len(intervals), windows

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

class MFCCExtractor(FeatureExtractor):

    def __init__(
            self,
            window_length,   # in seconds
            overlap_percent, # in percent
            frame_length,    # in seconds
            hop_length,      # in seconds
            sr=16000,
            n_fft=512,
            n_mfcc=20,
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
            self.n_mfcc=n_mfcc
            self.log_lift = 1e-6

    def as_features(self, path):
        _, windows = self.get_intervals(path)
        for w in windows:
            feature = librosa.feature.mfcc(
                w,
                sr=self.sr,
                n_fft=self.n_fft,
                n_mfcc=self.n_mfcc,
                win_length=self.frame_length,
                hop_length=self.hop_length
            )
            self.validate_numeric(feature)
            S = np.log10(feature + self.log_lift)
            if self.mean_normalize:
                S -= np.mean(S, axis=0)
            yield S
