#!/bin/bash

# make sure to have cuda 11.7 and gcc 9.0 installed
# install environment
pip install -r scripts/requirements.txt
sh scripts/download_prereq.sh

# download pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"

# download model stuff
wget http://audio2photoreal_models.berkeleyvision.org/PXB184_models.tar || { echo 'downloading model failed' ; exit 1; }
tar xvf PXB184_models.tar
rm PXB184_models.tar

# install rendering stuff
mkdir -p checkpoints/ca_body/data/
wget https://github.com/facebookresearch/ca_body/releases/download/v0.0.1-alpha/PXB184.tar.gz || { echo 'downloading ca body model failed' ; exit 1; }
tar xvf PXB184.tar.gz --directory checkpoints/ca_body/data/
rm PXB184.tar.gz