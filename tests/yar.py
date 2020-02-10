import yarrow
import os

dataset_root = os.getenv('DATASET_ROOT', '/home/ankit/Documents/github/datasets/')
test_csv_path = dataset_root + 'data/PUMS_california_demographics_1000/data.csv'


with yarrow.Analysis() as analysis:
    PUMS = yarrow.Dataset('PUMS', test_csv_path)

    #    age = PUMS[('age', int)]
    #    sex = PUMS[('sex', int)]
    #    inc = PUMS[('income', float)]


    mean_age = yarrow.dp_mean(
        PUMS[('income', float)],
        epsilon=4.0,
        minimum=0,
        maximum=100,
        num_records=1000
    )

print('starting releases')
vals = []
for x in range(5000):
    analysis.release()
    vals.append(analysis.release_proto.values[6].values['data'].f64.data[0])

print(vals)

