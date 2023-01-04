set -e
PATTERN=$1
OUTPUT_PATH=$2
IMGS_TO_CONCAT=$(ls $PATTERN | sort -V | xargs echo "concat:" |  tr ' ' '|' | sed 's/:|/:/g')
ffmpeg -i "$IMGS_TO_CONCAT" $OUTPUT_PATH
