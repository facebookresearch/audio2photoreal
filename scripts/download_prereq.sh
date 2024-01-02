
# install the prerequisite asset models (lip regressor and wav2vec)
wget http://audio2photoreal_models.berkeleyvision.org/asset_models.tar
tar xvf asset_models.tar
rm asset_models.tar

# we obtained the wav2vec models via these links:
# wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_large.pt -P ./assets/
# wget https://dl.fbaipublicfiles.com/fairseq/wav2vec/vq-wav2vec.pt -P ./assets/
