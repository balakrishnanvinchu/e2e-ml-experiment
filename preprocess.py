import pandas as pd
import joblib
import json

def preprocess_data(data, scaler_path='models/scaler.pkl', feature_names_path='models/feature_names.json'):
    """
    Preprocess input data for model prediction.
    
    Args:
        data (dict or pd.DataFrame): Input data with feature values
        scaler_path (str): Path to the saved scaler
        feature_names_path (str): Path to feature names JSON
    
    Returns:
        pd.DataFrame: Preprocessed data ready for prediction
    """
    # Load feature names
    with open(feature_names_path, 'r') as f:
        feature_info = json.load(f)
    feature_names = feature_info['features']
    
    # Convert to DataFrame if dict
    if isinstance(data, dict):
        data = pd.DataFrame([data])
    
    # Ensure all required features are present
    missing_features = set(feature_names) - set(data.columns)
    if missing_features:
        raise ValueError(f"Missing features: {missing_features}")
    
    # Select and order features
    data = data[feature_names]
    
    # Load scaler and transform
    scaler = joblib.load(scaler_path)
    data_scaled = scaler.transform(data)
    
    return pd.DataFrame(data_scaled, columns=feature_names)

if __name__ == "__main__":
    # Example usage
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
    
    processed = preprocess_data(sample_data)
    print("Preprocessed data shape:", processed.shape)
    print(processed.head())