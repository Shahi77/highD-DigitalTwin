#!/usr/bin/env bash
# usage: ./run_overfit_test.sh path/to/tracks.csv
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 path/to/tracks.csv"
  exit 1
fi
python3 train_and_eval.py "$1"
