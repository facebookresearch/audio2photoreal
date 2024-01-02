for i in "PXB184" "RLW104" "TXB805" "GQS883" 
do
    # download motion models
    wget http://audio2photoreal_models.berkeleyvision.org/${i}_models.tar || { echo 'downloading model failed' ; exit 1; }
    tar xvf ${i}_models.tar
    rm ${i}_models.tar
    
    # download ca body rendering checkpoints and assets
    mkdir -p checkpoints/ca_body/data/
    wget https://github.com/facebookresearch/ca_body/releases/download/v0.0.1-alpha/${i}.tar.gz || { echo 'downloading ca body model failed' ; exit 1; }
    tar xvf ${i}.tar.gz --directory checkpoints/ca_body/data/
    rm ${i}.tar.gz
done