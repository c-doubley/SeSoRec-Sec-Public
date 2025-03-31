#!/bin/bash
for k in {0..4}
do
    echo "Running fold $k"
    python Recommender.py $k &
    sleep 2
    python Social.py $k
    wait
done