from dataloader.db import DbCollection
from dataloader.download_reddit import download_reddit
from dataloader.download_pums import download_pums
from dataloader.make_sqlite import make_sqlite

download_reddit()
download_pums()
make_sqlite()

test_databases = DbCollection()
print(test_databases)