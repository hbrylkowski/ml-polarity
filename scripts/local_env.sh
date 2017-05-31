#!/usr/bin/env bash

SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"
DIR="$(dirname $SCRIPT)"
TRAIN_DATA="$(realpath $DIR/../data/train.csv)"
EVAL_DATA="$(realpath $DIR/../data/test.csv)"
MODEL_DIR="$(realpath $DIR/../)"
MODEL_PATH="$(realpath $DIR/../model.h5)"
