"""
Responsible for pulling one or several Librispeech urls into a local directory.
"""
import argparse
import os
import shutil
from torchaudio.datasets import LIBRISPEECH

def download(
    urls,
    path,
    overwrite
):
    """
    :param urls: A comma-separated string list of LibriSpeech urls to download.
    :type urls: str
    :param path: The absolute path to download the files into.
    :type path: str
    :param overwrite: Whether or not to overwrite the target directory.
    :type overwrite: str
    """
    abs_path = os.path.abspath(path)
    if overwrite:
        shutil.rmtree(path, ignore_errors=True)
    if not os.path.exists(path):
        os.mkdir(path)
    for url in urls.split(','):
        print(f'Writing Librispeech url: {url} to {path}')
        LIBRISPEECH(path, url, download=True)
        # remove the extraneous tar archive
        for f in filter(lambda x: x.endswith('tar.gz'), os.listdir(path)):
            if f.endswith('tar.gz'):
                os.remove(os.path.join(path, f))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--overwrite', help='Whether to overwrite the target directory [default: False]', action='store_true', default=False)
    parser.add_argument('--path', help='The relative / absolute path to serve as the root of the Librispeech files', type=str)
    parser.add_argument('--urls', help='A comma-separated list of Librispeech urls to download', type=str)
    args = parser.parse_args()
    download(args.urls, args.path, args.overwrite)
