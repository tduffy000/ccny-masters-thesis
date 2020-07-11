"""
The tf.keras.Model object formed from our config.yml representing our model
architecture.
"""
import tensorflow as tf

def get_dataset(feature_dir):
    pass

# https://arxiv.org/pdf/1803.05427.pdf
# https://towardsdatascience.com/tensorflow-speech-recognition-challenge-solution-outline-9c42dbd219c9
# https://towardsdatascience.com/debugging-a-machine-learning-model-written-in-tensorflow-and-keras-f514008ce736
# https://towardsdatascience.com/understanding-and-coding-a-resnet-in-keras-446d7ff84d33
class SpeakerVerificationModel(tf.keras.Model):

    def __init__(self):
        pass

    def call(self, inputs):
        return inputs

def main(args):
    dataset = get_dataset(args.feature_dir)
    model = SpeakerVerificationModel()
    model.compile()
    model.fit(dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--url', type=str)
    parser.add_argument('--feature_dir', type=str)
    parser.add_argument('--config_file', type=str)
    args = parser.parse_args()
    main(args)
