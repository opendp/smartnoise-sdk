import os

datasets = os.path.join(os.getcwd(), "../../../../../datasets")
datasets = os.path.abspath(datasets)

current = os.path.abspath(os.getcwd())

template = open('import.template')
t = template.read()
t = t.replace('{datasets}', datasets).replace('{current}', current)

with open('install.sql', 'w') as out:
    out.write(t)

# process for windows CR LF
pums = open(os.path.join(datasets, 'PUMS.csv'))
with open('PUMS.csv', 'w') as pums_out:
    pums_out.writelines(pums.readlines())

pums_large = open(os.path.join(datasets, 'PUMS_large.csv'))
with open('PUMS_large.csv', 'w') as pums_large_out:
    for line in pums_large.readlines():
        #line = ','.join(line.split(',')[1:])
        line = line.replace('""', "PersonID").replace('"','')
        pums_large_out.write(line)

pums_pid = open(os.path.join(datasets, 'PUMS_pid.csv'))
with open('PUMS_pid.csv', 'w') as pums_pid_out:
    pums_pid_out.writelines(pums_pid.readlines())

pums_dup = open(os.path.join(datasets, 'PUMS_dup.csv'))
with open('PUMS_dup.csv', 'w') as pums_dup_out:
    pums_dup_out.writelines(pums_dup.readlines())

pums_null = open(os.path.join(datasets, 'PUMS_null.csv'))
with open('PUMS_null.csv', 'w') as pums_null_out:
    pums_null_out.writelines(pums_null.readlines())
