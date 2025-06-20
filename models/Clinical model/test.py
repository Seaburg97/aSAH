"""
Clinical SVR Model Predictor
"""

import pandas as pd
import numpy as np
import joblib

def predict_mrs_outcome(model_path, age, hunt_hess_score, mfs_score, sebes_score):
    """
    Clinical mRS outcome predictor
    
    Args:
        model_path: Path to model file (.pkl or .joblib)
        age: Patient age in years
        hunt_hess_score: Hunt-Hess grade (1-5)
        mfs_score: Modified Fisher score (0-4)
        sebes_score: SEBES score (0-4)
    
    Returns:
        Dictionary with prediction results
    """
    
    try:
        model_data = joblib.load(model_path)
        
        # Get model components
        model = model_data['svr_model']
        scaler = model_data['scaler']
        encoders = model_data.get('encoders', {})
        feature_names = model_data['feature_names']
        mrs_threshold = model_data['mrs_threshold']
        
        # Build feature vector
        feature_values = {
            'Age': age, 'Hunt-Hess_score': hunt_hess_score,
            'mFS_score': mfs_score, 'SEBES_score': sebes_score
        }
        
        # Build full vector with defaults
        full_input = {}
        for feature in feature_names:
            if feature in feature_values:
                full_input[feature] = feature_values[feature]
            else:
                defaults = {
                    'Male': 0, 'Acute_hydrocephalus': 0, 'Posterior_circulation': 0,
                    'Size': 3.0, 'Localized_subarachnoid_hematoma': 0,
                    'Hypertension': 0, 'Clipping': 0
                }
                full_input[feature] = defaults.get(feature, 0)
        
        full_data = pd.DataFrame([full_input])
        
        # Apply encoders and scaling
        for column in full_data.columns:
            if column in encoders:
                try:
                    full_data[column] = encoders[column].transform(full_data[column].astype(str))
                except ValueError:
                    full_data[column] = -1
        
        X_scaled = scaler.transform(full_data)
        
        # Select used features
        all_used_features = model_data['all_used_features']
        selected_indices = [feature_names.index(feat) for feat in all_used_features]
        X_final = X_scaled[:, selected_indices]
        
        prediction = model.predict(X_final)[0]
        
        # Process results
        prediction_clipped = np.clip(prediction, 0, 6)
        binary_prediction = 1 if prediction_clipped > mrs_threshold else 0
        
        return {
            'mRS_prediction': round(prediction_clipped, 2),
            'binary_prediction': binary_prediction,
            'prognosis': 'Poor outcome' if binary_prediction == 1 else 'Good outcome',
            'mrs_threshold': mrs_threshold
        }
        
    except FileNotFoundError:
        return {'error': f'Model file not found: {model_path}'}
    except Exception as e:
        return {'error': f'Prediction failed: {str(e)}'}

def print_prediction_report(result):
    """Print a simplified prediction report"""
    if 'error' in result:
        print(f"❌ Prediction failed: {result['error']}")
        return
    
    print("🎯 mRS Outcome Prediction Report")
    print("=" * 45)
    print(f"📊 Predicted mRS Score: {result['mRS_prediction']}")
    print(f"🏷️ Prognosis: {result['prognosis']}")
    print(f"📏 Threshold: {result['mrs_threshold']}")
    print("=" * 45)

# Usage example
if __name__ == "__main__":
    print("🏥 Clinical SVR Model for mRS Prediction")
    print("=" * 45)
    
    # Test with clinical model
    model_path = 'Clinical_model.joblib'
    
    result = predict_mrs_outcome(
        model_path=model_path,
        age=61,
        hunt_hess_score=3,
        mfs_score=4,
        sebes_score=2
    )
    
    print_prediction_report(result)
    
    if 'error' not in result:
        print(f"\n🧪 Quick test with different values:")
        test_cases = [
            (35, 1, 1, 0),
            (75, 4, 4, 3),
            (65, 4, 4, 4)
        ]
        
        for age, hunt_hess, mfs, sebes in test_cases:
            result = predict_mrs_outcome(model_path, age, hunt_hess, mfs, sebes)
            if 'error' not in result:
                print(f"   Age {age}, H-H {hunt_hess}, mFS {mfs}, SEBES {sebes} → mRS: {result['mRS_prediction']}")