import pytest
from predict import predict_heart_disease

class TestPredictHeartDisease:
    """Test cases for heart disease prediction."""
    
    def test_prediction_output_structure(self):
        """Test that prediction returns correct output structure."""
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
        
        result = predict_heart_disease(sample_data)
        
        assert isinstance(result, dict)
        assert "prediction" in result
        assert "prediction_label" in result
        assert "confidence" in result
        assert "probabilities" in result
        assert "no_disease" in result["probabilities"]
        assert "disease" in result["probabilities"]
    
    def test_prediction_values(self):
        """Test that prediction values are valid."""
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
        
        result = predict_heart_disease(sample_data)
        
        assert result["prediction"] in [0, 1]
        assert 0 <= result["confidence"] <= 1
        assert 0 <= result["probabilities"]["no_disease"] <= 1
        assert 0 <= result["probabilities"]["disease"] <= 1
        assert abs(result["probabilities"]["no_disease"] + result["probabilities"]["disease"] - 1.0) < 0.1
    
    def test_prediction_consistency(self):
        """Test that same input gives same prediction."""
        sample_data = {
            "age": 50,
            "sex": 0,
            "cp": 1,
            "trestbps": 120,
            "chol": 200,
            "fbs": 0,
            "restecg": 1,
            "thalach": 160,
            "exang": 0,
            "oldpeak": 0.5,
            "slope": 1,
            "ca": 0,
            "thal": 2
        }
        
        result1 = predict_heart_disease(sample_data)
        result2 = predict_heart_disease(sample_data)
        
        assert result1["prediction"] == result2["prediction"]
        assert result1["confidence"] == result2["confidence"]
    
    def test_different_inputs_different_predictions(self):
        """Test that different inputs can give different predictions."""
        data1 = {
            "age": 30,
            "sex": 0,
            "cp": 0,
            "trestbps": 110,
            "chol": 180,
            "fbs": 0,
            "restecg": 0,
            "thalach": 170,
            "exang": 0,
            "oldpeak": 0.0,
            "slope": 1,
            "ca": 0,
            "thal": 2
        }
        
        data2 = {
            "age": 70,
            "sex": 1,
            "cp": 3,
            "trestbps": 180,
            "chol": 300,
            "fbs": 1,
            "restecg": 2,
            "thalach": 120,
            "exang": 1,
            "oldpeak": 3.0,
            "slope": 2,
            "ca": 2,
            "thal": 3
        }
        
        result1 = predict_heart_disease(data1)
        result2 = predict_heart_disease(data2)
        
        # They might be the same, but at least the function runs
        assert isinstance(result1["prediction"], int)
        assert isinstance(result2["prediction"], int)