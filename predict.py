import joblib
import json
import logging
from datetime import datetime
from preprocess import preprocess_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/predictions.log'),
        logging.StreamHandler()
    ]
)

def predict_heart_disease(data, model_path='models/best_model_logistic_regression.pkl'):
    """
    Predict heart disease risk from input data.
    
    Args:
        data (dict): Input features as dictionary
        model_path (str): Path to the saved model
    
    Returns:
        dict: Prediction results with class and probability
    """
    start_time = datetime.now()
    
    try:
        # Log input
        logging.info(f"Prediction request received: {json.dumps(data)}")
        
        # Preprocess the data
        processed_data = preprocess_data(data)
        
        # Load model
        model = joblib.load(model_path)
        
        # Make prediction
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0]
        
        # Get confidence (probability of positive class)
        confidence = probability[1] if prediction == 1 else probability[0]
        
        result = {
            "prediction": int(prediction),
            "prediction_label": "Heart Disease" if prediction == 1 else "No Heart Disease",
            "confidence": float(confidence),
            "probabilities": {
                "no_disease": float(probability[0]),
                "disease": float(probability[1])
            }
        }
        
        # Log result
        processing_time = (datetime.now() - start_time).total_seconds()
        logging.info(f"Prediction completed in {processing_time:.3f}s: {json.dumps(result)}")
        
        return result
        
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise

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
    
    result = predict_heart_disease(sample_data)
    print("Prediction Result:")
    print(json.dumps(result, indent=2))