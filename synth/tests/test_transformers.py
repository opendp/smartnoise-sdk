import numpy as np
import pandas as pd

from snsynth.factory.rdt.transformers.categorical import LabelEncodingTransformer, OneHotEncodingTransformer
from snsynth.factory.rdt.hyper_transformer import HyperTransformer


TEST_DATA_INDEX = [4, 6, 3, 8, 'a', 1.0, 2.0, 3.0]

def get_input_data():
    data = pd.DataFrame({
        'integer': [1, 2, 1, 3, 1, 4, 2, 3],
        'float': [0.1, 0.2, 0.1, np.nan, 0.1, 0.4, np.nan, 0.3],
        'categorical': ['a', 'a', np.nan, 'b', 'a', 'b', 'a', 'a'],
    }, index=TEST_DATA_INDEX)

    return data

def get_transformed_data():
    return pd.DataFrame({
        'integer.value': [1, 2, 1, 3, 1, 4, 2, 3],
        'float.value': [0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3],
        'float.is_null': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        'categorical.value': [0, 0, 1, 2, 0, 2, 0, 0],
    }, index=TEST_DATA_INDEX)

def get_label_encoded_data():
    return pd.DataFrame({
        'categorical.value': [0, 0, 1, 2, 0, 2, 0, 0],
        'integer.value': [1, 2, 1, 3, 1, 4, 2, 3],
        'float.value': [0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3],
        'float.is_null': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    }, index=TEST_DATA_INDEX)

def get_one_hot_encoded_data():
    return pd.DataFrame({
        'categorical.value0': [1, 1, 0, 0, 1, 0, 1, 1],
        'categorical.value1': [0, 0, 0, 1, 0, 1, 0, 0],
        'categorical.value2': [0, 0, 1, 0, 0, 0, 0, 0],
        'integer.value': [1, 2, 1, 3, 1, 4, 2, 3],
        'float.value': [0.1, 0.2, 0.1, 0.2, 0.1, 0.4, 0.2, 0.3],
        'float.is_null': [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
    }, index=TEST_DATA_INDEX)

def test_label_encoding_transformers():
    """
    Test LabelEncodingTransformer
    """
    # Setup
    field_transformers = {
        'categorical': LabelEncodingTransformer,
    }
    data = get_input_data()

    # Run
    ht = HyperTransformer(field_transformers=field_transformers)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed = ht.reverse_transform(transformed)

    # Assert
    expected_transformed = get_label_encoded_data()
    pd.testing.assert_frame_equal(transformed, expected_transformed)

    expected_reversed = get_input_data()
    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)

def test_one_hot_encoding_transformers():
    """
    Test LabelEncodingTransformer
    """
    # Setup
    field_transformers = {
        'categorical': OneHotEncodingTransformer,
    }
    data = get_input_data()

    # Run
    ht = HyperTransformer(field_transformers=field_transformers)
    ht.fit(data)
    transformed = ht.transform(data)
    reverse_transformed = ht.reverse_transform(transformed)
    
    # Assert
    expected_transformed = get_one_hot_encoded_data()
    pd.testing.assert_frame_equal(transformed, expected_transformed)

    expected_reversed = get_input_data()
    pd.testing.assert_frame_equal(expected_reversed, reverse_transformed)

def test_dtype_category():
    """Test that categorical variables of dtype category are supported."""
    # Setup
    data = pd.DataFrame({'a': ['a', 'b', 'c']}, dtype='category')

    # Run
    ht = HyperTransformer()
    ht.fit(data)
    transformed = ht.transform(data)
    reverse = ht.reverse_transform(transformed)

    # Assert
    pd.testing.assert_frame_equal(reverse, data)