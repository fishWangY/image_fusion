#!/usr/bin/env bash
source activate classify

cd ..
while true
do
    python -u eval.py
    sleep 50

done