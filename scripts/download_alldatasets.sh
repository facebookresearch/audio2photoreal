for i in "PXB184" "RLW104" "TXB805" "GQS883" 
do
    curl -L https://github.com/facebookresearch/audio2photoreal/releases/download/v1.0/${i}.zip -o ${i}.zip || { echo 'downloading dataset failed' ; exit 1; }
    unzip ${i}.zip -d dataset/
    rm ${i}.zip
done
