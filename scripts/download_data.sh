#!/bin/bash
echo "Downloading raw JSON data..."

URL_BASE="https://papertraildata.s3.us-west-1.amazonaws.com/"
FILENAMES=(
    #"raw_data.pkl"
    #"processed_data.pkl"
    #"processed_normalized_data.pkl"
    "hetero_data.pt"
    "hetero_data_no_coauthor.pt"
)




DATA_DIR=${1:-"./data"}

for FILENAME in "${FILENAMES[@]}"; do
    echo "Downloading $FILENAME..."
    wget "${URL_BASE}${FILENAME}" -O "${DATA_DIR}/${FILENAME}"
    echo "Downloaded $FILENAME to ${DATA_DIR}/${FILENAME}"
done