import argparse
import os
import time
import yaml

import tensorflow as tf

from dataset import GE2EDatasetLoader
from features import SpectrogramExtractor, MFCCExtractor
from loader import BatchLoader
from logger import logger
from loss import *
from model import SpeakerVerificationModel
from serializer import GE2ESpectrogramSerializer
from utils import *

def feature_engineering(conf):
    raw_data_conf = conf['raw_data']
    fe_conf = conf['features']
    fe_data_conf = conf['feature_data']

    extractor = SpectrogramExtractor(
        window_length=fe_conf['window_length'],     # in seconds
        overlap_percent=fe_conf['overlap_percent'], # in percent
        frame_length=fe_conf['frame_length'],       # in seconds
        hop_length=fe_conf['hop_length'],           # in seconds
        sr=16000,
        n_fft=512,
        n_mels=40,
        trim_top_db=30
    )

    BatchLoader(
        root_dir=raw_data_conf['path'],
        datasets=raw_data_conf['datasets'],
        target_dir=fe_data_conf['path'],
        feature_extractor=extractor,
        serializer=GE2ESpectrogramSerializer(),
        speakers_per_batch=fe_data_conf['speakers_per_batch'],
        utterances_per_speaker=fe_data_conf['utterances_per_speaker']
    ).load()

def train(conf, freeze=False):
    train_conf = conf['train']
    model_conf = train_conf['network']
    fe_data_conf = conf['feature_data']
    dataset_loader = GE2EDatasetLoader(fe_data_conf['path'])
    train_dataset = dataset_loader.get_train_dataset()
    dataset_metadata = dataset_loader.get_metadata()
    
    model = SpeakerVerificationModel(model_conf, dataset_metadata)
    optim = get_optimizer(
        type=model_conf['optimizer']['type'],
        lr=model_conf['optimizer']['lr'],
        momentum=model_conf['optimizer'].get('momentum', 0.0), # default for tf.keras.optimizers.{SGD, RMSprop}
        rho=model_conf['optimizer'].get('rho', 0.9),           # default for tf.keras.optimizers.RMSprop
        epsilon=model_conf['optimizer'].get('epsilon', 1e-7),
        clipnorm=model_conf['optimizer'].get('clipnorm', None)
    )

    model.compile(
        optimizer=optim,
        loss=get_embedding_loss(fe_data_conf['speakers_per_batch'], fe_data_conf['utterances_per_speaker'])
    )
    callbacks = []
    for callback, conf in model_conf['callbacks'].items():
        cb = get_callback(callback, conf, lr=model_conf['optimizer']['lr'], freeze=freeze)
        if cb is not None:
            callbacks.append(cb)
    model.fit(train_dataset, epochs=train_conf['epochs'], callbacks=callbacks)
    return dataset_metadata['feature_shape'], model

def freeze(model, input_shape, tflite=True):
    epoch_time = int(time.time())

    os.makedirs('frozen_models/full', exist_ok=True)
    model.save(f'frozen_models/full/{epoch_time}')

    os.makedirs('frozen_models/embedding', exist_ok=True)
    embedding_model = get_embedding_model(model, (None, input_shape[0], input_shape[1]))
    embedding_model.save(f'frozen_models/embedding/{epoch_time}')

    if tflite:
        converter = tf.lite.TFLiteConverter.from_saved_model(path)
        # https://www.tensorflow.org/lite/performance/model_optimization
        # https://www.tensorflow.org/lite/convert/rnn
        # https://www.tensorflow.org/lite/guide/roadmap
        # converter.optimizat
        tflite_model = converter.convert()
        os.makedirs(f'frozen_models/tiny/{epoch_time}', exist_ok=True)
        lite_path = f'frozen_models/tiny/{epoch_time}/model.tflite'
        logger.info(f'Converting trained model to TFLite in ./{lite_path}')
        with open(lite_path, 'wb') as f:
            f.write(tflite_model)

def load_model(path):
    return tf.keras.models.load_model(path)

def evaluate(conf, model):
    fe_data_conf = conf['feature_data']
    dataset_loader = GE2EDatasetLoader(fe_data_conf['path'])
    train_dataset, test_dataset = dataset_loader.get_datasets()
    N = fe_data_conf['speakers_per_batch']
    M = fe_data_conf['utterances_per_speaker']
    if test_dataset is not None:
        model.evaluate(test_dataset)

    write_eer_results(
        model=model,
        dataset_loader=dataset_loader,
        N=N,
        M=M,
        path=f'./testing_logs/{int(time.time())}'
    )

def main(args):
    model_epoch = int(time.time()) 

    assert(args.config_file is not None), 'Must specify a --config_file'

    with open(args.config_file, 'r') as stream:
        conf = yaml.safe_load(stream)
        stream.close()

    if args.feature_engineering:
        feature_engineering(conf)

    if args.train:
        if args.new_session:
            tf.keras.backend.clear_session()

        input_shape, model = train(conf, args.freeze_model)

        if args.freeze_model:
            freeze(model, input_shape, args.convert_to_lite)

    if args.test:
        model_path = args.test
        model = load_model(model_path)
        evaluate(conf, model)
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str, default=None)
    parser.add_argument('--feature_engineering', action='store_true', default=False)
    parser.add_argument('--new_session', action='store_true', default=True)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--freeze_model', action='store_true', default=False)
    parser.add_argument('--convert_to_lite', action='store_true', default=False)
    parser.add_argument('--test', type=str, default=None)
    args = parser.parse_args()
    main(args)
