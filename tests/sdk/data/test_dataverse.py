import os
import pytest
import pandas as pd

from opendp.smartnoise.data import dataverse_loader, dataverse_uri_loader


#  example dataset
#  https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/E9QKIZ/TRLS9U

@pytest.mark.parametrize("dataverse_details", [
        {"host": "https://dataverse.harvard.edu", "token": os.environ.get("SMARTNOISE_DATAVERSE_TEST_TOKEN"), "doi": "doi:10.7910/DVN/E9QKIZ/TRLS9U"},
        {"host": "https://dataverse.harvard.edu", "doi": "doi:10.7910/DVN/E9QKIZ/TRLS9U"},
        {"host": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/E9QKIZ/TRLS9U",},
        {"host": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/E9QKIZ/TRLS9U", "token": os.environ.get("SMARTNOISE_DATAVERSE_TEST_TOKEN"),}
    ])



# @pytest.mark.parametrize("dataverse_uri_details", "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/E9QKIZ/TRLS9U&amp;token=100")

def test_with_doi(dataverse_details):
    df = dataverse_loader(dataverse_details["host"], dataverse_details.get("doi"), dataverse_details.get("token"))
    assert df is not None

@pytest.mark.parametrize("dataverse_endpoint", [{"uri": "https://dataverse.harvard.edu/api/access/datafile/:persistentId?persistentId=doi:10.7910/DVN/E9QKIZ/TRLS9U&token=100"}])
def test_with_uri(dataverse_endpoint):
    df = dataverse_uri_loader(dataverse_endpoint["uri"])
    assert df is not None and isinstance(df, pd.DataFrame) and not df.empty
