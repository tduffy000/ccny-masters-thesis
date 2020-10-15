import argparse
import yaml
import os
import time
import tensorflow as tf
from logger import logger
from dataset import GE2EDatasetLoader
from transformer import GE2EBatchLoader
from model import SpeakerVerificationModel
from utils import get_callback, get_optimizer

def feature_engineering(conf):
    raw_data_conf = conf['raw_data']
    fe_conf = conf['features']
    fe_data_conf = conf['feature_data']

    GE2EBatchLoader(
        root_dir=raw_data_conf['path'],
        datasets=raw_data_conf['datasets'],
        target_dir=fe_data_conf['path'],
        train_split=fe_data_conf['train_split'],
        tfrecord_examples_per_file=fe_data_conf['tfrecord_examples_per_file'],
        window_length=fe_conf['window_length'], # in seconds
        overlap_percent=fe_conf['overlap_percent'], # in percent
        frame_length=fe_conf['frame_length'], # in seconds
        hop_length=fe_conf.get('hop_length', -1), # in seconds
        n_fft=fe_conf.get('n_fft', -1),
        n_mels=fe_conf.get('n_mels', -1),
        sr=conf['sr'],
        trim_top_db=fe_conf['trim_top_db'],
        speakers_per_batch=fe_data_conf['speakers_per_batch'],
        utterances_per_speaker=fe_data_conf['utterances_per_speaker']
    ).load()

def train(conf, freeze=False):
    train_conf = conf['train']
    model_conf = train_conf['network']
    fe_data_conf = conf['feature_data']
    dataset_loader = GE2EDatasetLoader(
        fe_data_conf['path'],
        fe_data_conf['batch_size']
    )
    train_dataset = dataset_loader.get_train_dataset()
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
        loss='sparse_categorical_crossentropy',
        metrics=[
            'accuracy'
        ]
    )
    callbacks = []
    for callback, conf in model_conf['callbacks'].items():
        cb = get_callback(callback, conf, lr=model_conf['optimizer']['lr'], freeze=freeze)
        if cb is not None:
            callbacks.append(cb)
    model.fit(train_dataset, epochs=train_conf['epochs'], callbacks=callbacks)
    logger.info('Finished training, now evaluating...')
    return model

def evaluate(conf, model):
    fe_data_conf = conf['feature_data']
    dataset_loader = DatasetLoader(fe_data_conf['path'], fe_data_conf['batch_size'])
    test_dataset = dataset_loader.get_test_dataset()
    model.evaluate(test_dataset)

def freeze(model, tflite=True):
    epoch_time = int(time.time())
    os.makedirs('frozen_models/full', exist_ok=True)
    path = f'frozen_models/full/{epoch_time}'
    logger.info(f'Freezing trained model to ./{path}')
    model.save(path)
    if tflite:
        converter = tf.lite.TFLiteConverter.from_saved_model(path)
        tflite_model = converter.convert()
        os.makedirs(f'frozen_models/tiny/{epoch_time}', exist_ok=True)
        lite_path = f'frozen_models/tiny/{epoch_time}/model.tflite'
        logger.info(f'Converting trained model to TFLite in ./{lite_path}')
        with open(lite_path, 'wb') as f:
            f.write(tflite_model)

def main(args):
    assert(args.config_file is not None), 'Must specify a --config_file'

    with open(args.config_file, 'r') as stream:
        conf = yaml.safe_load(stream)
        stream.close()

    if args.feature_engineering:
        feature_engineering(conf)

    if args.train:
        if args.new_session:
            tf.keras.backend.clear_session()

        model = train(conf, args.freeze_model)
        evaluate(conf, model)

        if args.freeze_model:
            freeze(model, args.convert_to_lite)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--feature_engineering', action='store_true', default=False)
    parser.add_argument('--new_session', action='store_true', default=True)
    parser.add_argument('--freeze_model', action='store_true', default=False)
    parser.add_argument('--convert_to_lite', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
