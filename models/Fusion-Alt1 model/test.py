"""
Simple SVR Model Predictor - Single Prediction Only
Uses extracted standardization parameters for accurate prediction
"""

import pandas as pd
import numpy as np
import joblib
class OptimizedPredictor:
    """
    OptimizedPredictor class that matches the original model structure
    """
    def __init__(self, core_engine=None, *args, **kwargs):
        self._core_engine = core_engine
        for key, value in kwargs.items():
            setattr(self, key, value)
        if args and core_engine is None:
            self._core_engine = args[0]
    
    def predict(self, X):
        if hasattr(self, '_core_engine') and self._core_engine is not None:
            return self._core_engine.predict(X)
        raise AttributeError("No working predict method found in OptimizedPredictor")
def predict_mrs_outcome(model_path, age, hunt_hess_score, mfs_score, sebes_score, prediction_pre):
    """
    Predict mRS outcome using the trained SVR model with correct standardization parameters
    
    Args:
        model_path: Path to the trained model file
        age: Patient age in years
        hunt_hess_score: Hunt-Hess grade (1-5)
        mfs_score: Modified Fisher score (0-4)
        sebes_score: SEBES score (0-4)
        prediction_pre: Pre-operative prediction value (0-6 range)
    
    Returns:
        Dictionary with prediction results
    """
    
    # Load the trained model
    model_data = joblib.load(model_path)
    
    model = model_data['model']
    encoders = model_data['encoders']
    scaler = model_data['scaler']
    all_used_features = model_data['all_used_features']
    feature_names = model_data['feature_names']
    mrs_threshold = model_data['mrs_threshold']
    
    # Extracted standardization parameters for Prediction_pre
    PREDICTION_PRE_MEAN = 2.2823532003
    PREDICTION_PRE_STD = 1.3692531308
    
    # Step 1: Build complete feature vector for scaler (11 features)
    # Must match the exact order from training: feature_names
    feature_values = {
        'Age': age,
        'Hunt-Hess_score': hunt_hess_score,
        'mFS_score': mfs_score,
        'SEBES_score': sebes_score,
        'Male': 0,                                    # Default: female
        'Acute_hydrocephalus': 0,                    # Default: no hydrocephalus
        'Posterior_circulation': 0,                   # Default: anterior circulation
        'Size': 5.0,                                 # Default: medium size
        'Localized_subarachnoid_hematoma': 0,        # Default: no hematoma
        'Hypertension': 0,                           # Default: no hypertension
        'Clipping': 0                                # Default: no clipping
    }
    
    # Create DataFrame in the exact order from training
    ordered_data = []
    for feature_name in feature_names:
        if feature_name in feature_values:
            ordered_data.append(feature_values[feature_name])
        else:
            ordered_data.append(0)  # Default value for any missing feature
    
    # Convert to DataFrame with correct column order
    full_data = pd.DataFrame([ordered_data], columns=feature_names)
    
    # Step 2: Apply encoders (no categorical features in this case, but keeping for completeness)
    for column in full_data.columns:
        if column in encoders:
            try:
                full_data[column] = encoders[column].transform(full_data[column].astype(str))
            except ValueError:
                full_data[column] = -1  # Handle unseen categories
    
    # Step 3: Standardize the complete feature vector using the trained scaler
    X_scaled_full = scaler.transform(full_data)
    
    # Step 4: Extract the 4 Lasso-selected features in correct order
    # Based on training results: ['Hunt-Hess_score', 'mFS_score', 'SEBES_score', 'Age']
    final_features = []
    lasso_selected_features = ['Hunt-Hess_score', 'mFS_score', 'SEBES_score', 'Age']
    
    for feat in lasso_selected_features:
        if feat in feature_names:
            idx = feature_names.index(feat)
            final_features.append(X_scaled_full[0, idx])
    
    # Step 5: Add the additional feature (Prediction_pre)
    # This feature was added separately and needs manual standardization
    prediction_pre_scaled = (prediction_pre - PREDICTION_PRE_MEAN) / PREDICTION_PRE_STD
    final_features.append(prediction_pre_scaled)
    
    # Step 6: Make prediction
    X_final = np.array(final_features).reshape(1, -1)
    prediction = model.predict(X_final)[0]
    prediction_clipped = np.clip(prediction, 0, 6)
    binary_prediction = 1 if prediction_clipped > mrs_threshold else 0
    
    return {
        'mRS_prediction': round(prediction_clipped, 2),
        'binary_prediction': binary_prediction,
        'prognosis': 'Poor outcome' if binary_prediction == 1 else 'Good outcome',
        'risk_level': get_risk_level(prediction_clipped),
        'mrs_threshold': mrs_threshold,
        'confidence': get_confidence_level(prediction_clipped, mrs_threshold)
    }

def get_risk_level(mrs_score):
    """Determine risk level based on mRS score"""
    if mrs_score <= 2:
        return "Low risk"
    elif mrs_score <= 4:
        return "Medium risk"
    else:
        return "High risk"

def get_confidence_level(prediction, threshold):
    """Calculate prediction confidence based on distance from threshold"""
    distance = abs(prediction - threshold)
    if distance >= 1.5:
        return "High confidence"
    elif distance >= 0.8:
        return "Medium confidence"
    else:
        return "Low confidence"

def print_prediction_report(result):
    """Print a detailed prediction report"""
    if 'error' in result:
        print(f"❌ Prediction failed: {result['error']}")
        return
    
    print("🎯 mRS Outcome Prediction Report")
    print("=" * 45)
    print(f"📊 Predicted mRS Score: {result['mRS_prediction']}")
    print(f"🏷️ Prognosis: {result['prognosis']}")
    print(f"⚠️ Risk Level: {result['risk_level']}")
    print(f"🎯 Confidence: {result['confidence']}")
    print(f"📏 Threshold: {result['mrs_threshold']}")
    print("-" * 45)
    
    # Clinical interpretation
    mrs_score = result['mRS_prediction']
    print("📋 Clinical Interpretation:")
    if mrs_score <= 1:
        print("   • No significant disability")
        print("   • Able to carry out all usual activities")
    elif mrs_score <= 2:
        print("   • Slight disability")
        print("   • Able to look after own affairs")
    elif mrs_score <= 3:
        print("   • Moderate disability")
        print("   • Requires some help, but able to walk unassisted")
    elif mrs_score <= 4:
        print("   • Moderately severe disability")
        print("   • Unable to attend to bodily needs without assistance")
    elif mrs_score <= 5:
        print("   • Severe disability")
        print("   • Requires constant nursing care and attention")
    else:
        print("   • Death")
    
    print("=" * 45)

# Usage example
if __name__ == "__main__":
    print("🏥 Simple SVR Model for mRS Outcome Prediction")
    print("=" * 60)
    print("Using extracted standardization parameters:")
    print("  Prediction_pre Mean: 2.2823532003")
    print("  Prediction_pre Std:  1.3692531308")
    
    model_path = 'Fusion-Alt1_model.pkl'  #modelname
    
    # Example: Single patient prediction
    print("\n📋 Example: Single Patient Prediction")
    print("-" * 40)
    
    result = predict_mrs_outcome(
        model_path=model_path,
        age=61,                        # 61 years old
        hunt_hess_score=3,             # Hunt-Hess grade 3
        mfs_score=4,                   # Modified Fisher score 4
        sebes_score=2,                 # SEBES score 2
        prediction_pre=2.5             # Pre-operative prediction
    )
    
    print_prediction_report(result)
    
    print(f"\n✅ Predictor ready to use with correct standardization parameters!")
    print(f"🔧 Usage:")
    print(f"   result = predict_mrs_outcome(model_path, age, hunt_hess, mfs, sebes, prediction_pre)")
    print(f"   print(result['mRS_prediction'])")
    
    # Quick test with different values
    print(f"\n🧪 Quick test with different values:")
    test_cases = [
        (35, 1, 1, 0, 1.5),
        (75, 4, 4, 3, 3.8),
        (58, 3, 2, 1, 2.2)
    ]
    
    for age, hunt_hess, mfs, sebes, prediction_pre in test_cases:
        result = predict_mrs_outcome(model_path, age, hunt_hess, mfs, sebes, prediction_pre)
        print(f"   Age {age}, H-H {hunt_hess}, mFS {mfs}, SEBES {sebes}, Pred_pre {prediction_pre} → mRS: {result['mRS_prediction']}")