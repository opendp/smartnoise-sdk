#!/bin/bash
DATASETS=../../../../dp-test-datasets/data
PUMS=${DATASETS}/PUMS_california_demographics_1000/data.csv
PUMS_large=${DATASETS}/PUMS_california_demographics/data.csv
cat $PUMS | tail -n +2 > PUMS.csv
cat $PUMS_large | tail -n +2 > PUMS_large.csv