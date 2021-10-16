# set DATA to location of dp-test-datasets/data/
DATASETS=../../../../../datasets/
PUMS=${DATASETS}PUMS.csv
PUMS_LARGE=${DATASETS}PUMS_large.csv
PUMS_PID=${DATASETS}PUMS_pid.csv
PUMS_DUP=${DATASETS}PUMS_dup.csv
tail -n +2 $PUMS > PUMS.csv
tail -n +2 $PUMS_LARGE > PUMS_large.csv
tail -n +2 $PUMS_PID > PUMS_pid.csv
tail -n +2 $PUMS_DUP > PUMS_dup.csv
ls -l
psql -h localhost -U postgres < pums.sql
echo "Done!"
rm PUMS.csv
rm PUMS_pid.csv
rm PUMS_dup.csv
rm PUMS_large.csv
