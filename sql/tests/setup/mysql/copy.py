import os
import subprocess

git_root_dir = subprocess.check_output("git rev-parse --show-toplevel".split(" ")).decode("utf-8").strip()
source_folder = os.path.abspath(os.path.join(git_root_dir, 'datasets'))
dest_folder = '/var/lib/mysql-files/'

for file in ['PUMS.csv', 'PUMS_dup.csv', 'PUMS_large.csv', 'PUMS_pid.csv']:
    source_path = os.path.join(source_folder, file)
    dest_path = os.path.join(dest_folder, file)
    os.system('cp {} {}'.format(source_path, dest_path))

source_path = os.path.join(source_folder, 'PUMS_null.csv')
dest_path = os.path.join(dest_folder, 'PUMS_null.csv')

with open(source_path, 'r') as f:
    lines = f.read().splitlines(keepends=False)
    with open(dest_path, 'w') as f2:
        for line in lines:
            line_parts = line.split(',')
            line_parts = [p if p != '' else "\\N" for p in line_parts]
            f2.write(','.join(line_parts))
            f2.write('\n')

