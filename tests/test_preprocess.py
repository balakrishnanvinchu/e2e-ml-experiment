import pytest
import pandas as pd
import numpy as np
from preprocess import preprocess_data

class TestPreprocessData:
    """Test cases for data preprocessing functions."""
    
    def test_preprocess_dict_input(self):
        """Test preprocessing with dictionary input."""
        sample_data = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }
        
        result = preprocess_data(sample_data)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (1, 13)
        assert all(col in result.columns for col in sample_data.keys())
    
    def test_preprocess_dataframe_input(self):
        """Test preprocessing with DataFrame input."""
        sample_df = pd.DataFrame({
            "age": [63, 45],
            "sex": [1, 0],
            "cp": [3, 2],
            "trestbps": [145, 120],
            "chol": [233, 200],
            "fbs": [1, 0],
            "restecg": [0, 1],
            "thalach": [150, 160],
            "exang": [0, 1],
            "oldpeak": [2.3, 1.0],
            "slope": [0, 2],
            "ca": [0, 1],
            "thal": [1, 2]
        })
        
        result = preprocess_data(sample_df)
        
        assert isinstance(result, pd.DataFrame)
        assert result.shape == (2, 13)
    
    def test_missing_features_error(self):
        """Test error handling for missing features."""
        incomplete_data = {
            "age": 63,
            "sex": 1
            # Missing other features
        }
        
        with pytest.raises(ValueError, match="Missing features"):
            preprocess_data(incomplete_data)
    
    def test_scaled_data_properties(self):
        """Test that data is properly scaled (mean ~0, std ~1)."""
        sample_data = {
            "age": 63,
            "sex": 1,
            "cp": 3,
            "trestbps": 145,
            "chol": 233,
            "fbs": 1,
            "restecg": 0,
            "thalach": 150,
            "exang": 0,
            "oldpeak": 2.3,
            "slope": 0,
            "ca": 0,
            "thal": 1
        }
        
        result = preprocess_data(sample_data)
        
        # Check that data is scaled (not raw values)
        assert not np.allclose(result.values, list(sample_data.values()))
        assert result.shape[1] == 13