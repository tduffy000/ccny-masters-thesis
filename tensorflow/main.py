import argparse
import yaml
import os
import time
import tensorflow as tf
from logger import logger
from dataset import DatasetLoader
from transformer import FeatureLoader
from model import SpeakerVerificationModel
from utils import get_callback, get_optimizer

# TODO: we could probably put these into a Config object
def main(args):
    with open(args.config_file, 'r') as stream:
        conf = yaml.safe_load(stream)
        stream.close()

    raw_data_conf = conf['raw_data']
    raw_data_path = raw_data_conf['path']
    feature_data_path = conf['feature_data']['path']
    if args.feature_engineering:
        fe_conf = conf['features']
        logger.info(f'Running feature engineering with config: {fe_conf}')
        FeatureLoader(
            root_dir=raw_data_path,
            datasets=raw_data_conf['datasets'],
            target_dir=feature_data_path,
            window_length=fe_conf['window_length'], # in seconds
            overlap_percent=fe_conf['overlap_percent'], # in percent
            frame_length=fe_conf['frame_length'], # in seconds
            hop_length=fe_conf['hop_length'], # in seconds
            buffer_flush_size=conf['feature_data']['buffer_flush_size'], # in features
            n_fft=fe_conf['n_fft'],
            n_mels=fe_conf['n_mels'],
            sr=conf['sr'],
            use_preemphasis=fe_conf['use_preemphasis'],
            preemphasis_coef=fe_conf.get('preemphasis_coef', 0.97),
            trim_silence=fe_conf['trim_silence'],
            trim_top_db=fe_conf['trim_top_db'],
            normalization=fe_conf.get('normalization', None),
            test_data_ratio=conf['feature_data']['test_ratio']
        ).load()
    if args.train:
        if args.new_session:
            tf.keras.backend.clear_session()
        train_conf = conf['train']
        dataset_loader = DatasetLoader(
            root_dir=feature_data_path,
            batch_size=train_conf['batch_size'],
            example_dim=train_conf['input_dimensions']
        )
        train_dataset, test_dataset = dataset_loader.get_dataset()
        model_conf = train_conf['network']
        dataset_metadata = dataset_loader.get_metadata()
        model = SpeakerVerificationModel(model_conf, dataset_metadata)
        optim = get_optimizer(
            type=model_conf['optimizer']['type'],
            lr=model_conf['optimizer']['lr'],
            momentum=model_conf['optimizer'].get('momentum', 0.0), # default for tf.keras.optimizers.{SGD, RMSprop}
            rho=model_conf['optimizer'].get('rho', 0.9), # default for tf.keras.optimizers.RMSprop
            epsilon=model_conf['optimizer'].get('epsilon', 1e-7)
        )
        model.compile(
            optimizer=optim,
            loss=model_conf['loss'],
            metrics=['accuracy'] # add metrics for the embeddings & separation
        )
        callbacks = []
        for callback, conf in model_conf['callbacks'].items():
            # TODO: clean up lr flag here
            callbacks.append(get_callback(callback, conf, lr=model_conf['optimizer']['lr']))
        model.fit(train_dataset, epochs=train_conf['epochs'], callbacks=callbacks)
        logger.info('Finished training, now evaluating...')
        model.evaluate(test_dataset)
        if args.freeze_model:
            path = f'frozen_models/{int(time.time())}'
            logger.info(f'Freezing trained model to ./{path}')
            if not os.path.exists('frozen_models'):
                os.makedirs('frozen_models')
            model.save(path)

    # if args.test:
    #     # load validation shards

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--feature_engineering', action='store_true', default=False)
    parser.add_argument('--new_session', action='store_true', default=True)
    parser.add_argument('--freeze_model', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
