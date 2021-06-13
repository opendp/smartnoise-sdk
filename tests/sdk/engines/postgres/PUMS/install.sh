# set this to location of dp-test-datasets
DATA=../../../../../../dp-test-datasets/data/

PUMS=${DATA}PUMS_california_demographics_1000/data.csv
PUMS_LARGE=${DATA}PUMS_california_demographics/data.csv
tail -n +2 $PUMS > PUMS.csv
tail -n +2 $PUMS_LARGE > PUMS_large.csv
psql -U postgres < pums.sql
rm PUMS.csv
rm PUMS_large.csv