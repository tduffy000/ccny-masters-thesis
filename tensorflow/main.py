import argparse
import yaml
from transformer import FeatureLoader

def main(args):
    with open(args.config_file, 'r') as stream:
        conf = yaml.safe_load(stream)
        stream.close()

    url = conf['url']
    raw_data_path = conf['raw_data']['path']
    feature_data_path = conf['feature_data']['path']
    if args.feature_engineering:
        fe_conf = conf['features']
        FeatureLoader(
            source_dir=raw_data_path,
            target_dir=feature_data_path,
            url=url,
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
            normalization=fe_conf.get('normalization', None)
        ).load()
    if args.train:
        train_conf = conf['train']
        dataset = DatasetLoader(
            source_dir=feature_data_path,
            batch_size=train_conf['batch_size'],
            example_dim=train_conf['input_dimenions']
        ).load_dataset()
        for model_conf in train_conf['models']:
            model = None
            model.fit(dataset, epochs=train_conf['epochs'])
            # store results

    # if args.test:
    #     # load validation shards

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_file', type=str)
    parser.add_argument('--feature_engineering', action='store_true', default=False)
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    main(args)
