#!/bin/bash

while getopts ":p:u:" opt; do
	case $opt in
		p) path="$OPTARG"
		;;
		u) url="$OPTARG"
		;;
		\?) echo "Only -p (path) and -u (url) are valid args" >&2
		;;
	esac
done

printf "Copying $url into $path [will overwrite!]\n"
if [ -d $path/LibriSpeech/$url ]; then
	rm -rf $path/LibriSpeech/$url
fi
librispeech_url=http://openslr.org/resources/12
wget $librispeech_url/$url.tar.gz -P $path
cd $path
tar -xvf $path/$url.tar.gz
rm $path/$url.tar.gz
