#!/bin/bash
echo "Running five-fold cross validation"
python Recommender.py &
sleep 2
python Social.py
wait