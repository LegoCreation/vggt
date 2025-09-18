#!/bin/bash

BASE_DATASET_DIR=/storage/group/dataset_mirrors/uco3d/uco3d_preprocessed_new
DESTINATION_DIR=/storage/slurm/schnaus/dl4sai_2025/uco3d

mkdir -p $DESTINATION_DIR
USER_NAME="$(whoami)"
GROUP_NAME="$(id -gn)"
WORKERS=8
export DESTINATION_DIR

find "$BASE_DATASET_DIR" -mindepth 1 -maxdepth 1 -type d | \
  parallel -j"$WORKERS" --bar '
  echo ">>> Copying directory: {}"
  rsync -ahv --whole-file --info=progress2 --inplace --no-compress --partial \
        --chown="$USER_NAME:$GROUP_NAME" {} "$DESTINATION_DIR"/
'

