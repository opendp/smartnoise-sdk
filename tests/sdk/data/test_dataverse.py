import os
import pytest

from opendp.whitenoise.data import dataverse_loader


#  example dataset
#  https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/E9QKIZ/TRLS9U

@pytest.mark.parametrize("dataverse_details", [
        {"host": "https://dataverse.harvard.edu", "token": os.environ.get("WHITENOISE_DATAVERSE_TEST_TOKEN"), "doi": "doi:10.7910/DVN/E9QKIZ/TRLS9U"},
        {"host": "https://dataverse.harvard.edu", "doi": "doi:10.7910/DVN/E9QKIZ/TRLS9U"},
        {"host": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/E9QKIZ/TRLS9U",},
        {"host":
         "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/E9QKIZ/TRLS9U", "token": os.environ.get("WHITENOISE_DATAVERSE_TEST_TOKEN"),}
    ])
def test_with_doi(dataverse_details):
    df = dataverse_loader(dataverse_details["host"], dataverse_details.get("doi"), dataverse_details.get("token"))
    assert df is not None
