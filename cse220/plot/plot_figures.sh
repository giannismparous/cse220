#!/bin/bash
set -x #echo on
cd "$(dirname "$0")"

OUTPUT_DIR=/home/$USER/plot/$EXPERIMENT
SIM_PATH=/home/$USER/exp/simulations
DESCRIPTOR_PATH=/home/$USER/$EXPERIMENT.json

mkdir -p $OUTPUT_DIR

if [ "$EXPERIMENT" = "lab2" ]; then
  python3 plot_lab2.py -o $OUTPUT_DIR -d $DESCRIPTOR_PATH -s $SIM_PATH
else
  python3 plot_lab1.py -o $OUTPUT_DIR -d $DESCRIPTOR_PATH -s $SIM_PATH
fi
