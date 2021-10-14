from dataloader import DbCollection, download_data_files
download_data_files()
test_databases = DbCollection()
print(test_databases)