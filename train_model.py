from ml_model import EmergencyPredictor

def main():
    print("=" * 40)
    print("Starting Emergency Response ML Model Training")
    print("=" * 40)
    
    predictor = EmergencyPredictor()
    
    print("Training new model...")
    accuracy = predictor.train_model()
    
    print("\n✅ Model training completed!")
    print(f"   Final accuracy: {accuracy:.2f}")
    print(f"   Model saved to: {predictor.model_path}")
    
    print("\n--- Testing model with a sample high-risk patient ---")
    sample_features = [75, 1, 130, 85, 55, 92, 99.5, 25]
    result = predictor.predict(sample_features)
    print(f"   Prediction -> Needs ICU: {result['needs_icu']}, Risk Score: {result['risk_score']:.2f}")
    print("=" * 40)

if __name__ == "__main__":
    main()