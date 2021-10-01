# set DATA to location of dp-test-datasets/data/
DATASETS=../../../../../datasets/
sqlcmd -S localhost -P ${SA_PASSWORD} -U sa -i pums.sql
PUMS=${DATA}PUMS_california_demographics_1000/data.csv
PUMS_LARGE=${DATA}PUMS_california_demographics/data.csv
PUMS_PID=${DATASETS}PUMS_pid.csv
PUMS_DUP=${DATASETS}PUMS_dup.csv
tail -n +2 $PUMS > PUMS.csv
tail -n +2 $PUMS_LARGE > PUMS_large.csv
tail -n +2 $PUMS_PID > PUMS_pid.csv
tail -n +2 $PUMS_DUP > PUMS_large.csv
psql -h $POSTGRES_HOST -U postgres < pums.sql
rm PUMS.csv
rm PUMS_pid.csv
rm PUMS_dup.csv
rm PUMS_large.csv
