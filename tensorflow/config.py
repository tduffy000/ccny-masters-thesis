import json
import yaml

class Config:

    def __init__(self):
        pass

    def as_dict(self):
        return vars(self)

    def to_json(self, path):
        d = vars(self)
        with open(path, 'w') as stream:
            json.dump(d, path)
        return

class RawDataConfig(Config):

    def __init__(self, path):
        super(RawDataConfig, self).__init__()
        self.path = path

class FeatureConfig(Config):

    def __init__(
        self,
        path,
        type,
        window_length,
        overlap_percent,
        frame_length, # seconds
        hop_length, # seconds
        n_fft,
        n_mels,
        use_preemphasis,
        trim_silence,
        trim_top_db,
        normalization
    ):
        super(FeatureConfig, self).__init__()
        self.path = path
        self.type = type
        self.window_length = window_length
        self.overlap_percent = overlap_percent
        self.frame_length = frame_length, # seconds
        self.hop_length = hop_length, # seconds
        self.n_fft = n_fft,
        self.n_mels = n_mels,
        self.use_preemphasis = use_preemphasis,
        self.trim_silence = trim_silence,
        self.trim_top_db = trim_top_db,
        self.normalization = normalization

class TrainingConfig(Config):

    def __init__(
        self,
        epochs,
        batch_size,
        input_dimensions,
        network_config
    ):
        super(TrainingConfig, self).__init__()
        self.epochs = epochs
        self.batch_size = batch_size
        self.input_dimensions = input_dimensions
        self.network_config = network_config # populates ModelConfig


class ValidationConfig(Config):

    def __init__():
        super(ValidationConfig, self).__init__()

class ModelConfig(Config):

    def __init__(self):
        super(ModelConfig, self).__init__()
