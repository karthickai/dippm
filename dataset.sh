#!/bin/bash

URL="https://github.com/karthickai/dippm/releases/download/v1.0.0/dataset_v1.tar"

DATA_FOLDER="data"

if [ ! -d "$DATA_FOLDER" ]; then
  mkdir $DATA_FOLDER
fi

curl  $URL -o dataset.tar
tar -xvf dataset.tar -C $DATA_FOLDER
