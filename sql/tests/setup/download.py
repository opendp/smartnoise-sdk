from dataloader import TestDbCollection, download_data_files
download_data_files()
test_databases = TestDbCollection()
print(test_databases)