for i in "PXB184" "RLW104" "TXB805" "GQS883" 
do
    wget <TODO_URL>${i}.tar || { echo 'downloading dataset failed' ; exit 1; }
    tar xvf ${i}.tar --directory dataset/
    rm ${i}.tar
done