#!/usr/bin/env python3
"""
Make Predictions with Trained MTBM Model
=========================================

This script uses your trained model to predict advance rates.

Usage:
    python make_predictions.py

Requirements:
    - trained_advance_rate_model.pkl (created by train_with_real_data.py)
"""

import pandas as pd
import numpy as np
import joblib
import sys

def load_model(model_file='trained_advance_rate_model.pkl'):
    """Load the trained model"""

    try:
        model_package = joblib.load(model_file)

        print("=" * 70)
        print("📂 MODEL LOADED SUCCESSFULLY")
        print("=" * 70)
        print(f"\nModel: {model_package['model_name']}")
        print(f"Training Date: {model_package['training_date']}")
        print(f"Test R² Score: {model_package['performance']['test_r2']:.3f}")
        print(f"RMSE: {model_package['performance']['rmse']:.2f} m/day")

        print(f"\nFeatures required:")
        for i, feat in enumerate(model_package['feature_columns'], 1):
            feat_display = feat.replace('geo_encoded', 'geological_type')
            print(f"   {i}. {feat_display}")

        print(f"\nAvailable geological types:")
        for geo in model_package['geo_mapping'].keys():
            print(f"   • {geo}")

        return model_package

    except FileNotFoundError:
        print("=" * 70)
        print("❌ ERROR: Model file not found!")
        print("=" * 70)
        print(f"\nCould not find: {model_file}")
        print("\nPlease run 'python train_with_real_data.py' first to train the model.")
        sys.exit(1)


def predict_single(geological_type, thrust_force, torque, rpm,
                   earth_pressure=None, advance_speed=None):
    """
    Predict advance rate for a single scenario

    Args:
        geological_type: Type of ground (e.g., 'soft_clay', 'dense_sand')
        thrust_force: Thrust force in kN
        torque: Torque in kN·m
        rpm: Rotation speed (RPM)
        earth_pressure: Earth pressure in bar (optional)
        advance_speed: Advance speed in mm/min (optional)

    Returns:
        Predicted advance rate in m/day
    """

    # Load model
    model_package = load_model()

    model = model_package['model']
    features = model_package['feature_columns']
    geo_mapping = model_package['geo_mapping']

    # Check geological type
    if geological_type not in geo_mapping:
        print(f"\n❌ Error: Unknown geological type '{geological_type}'")
        print(f"\nAvailable types: {list(geo_mapping.keys())}")
        return None

    # Prepare input
    input_data = {
        'geo_encoded': geo_mapping[geological_type],
        'thrust_force': thrust_force,
        'torque': torque,
        'rpm': rpm
    }

    # Add optional features
    if earth_pressure is not None and 'earth_pressure' in features:
        input_data['earth_pressure'] = earth_pressure

    if advance_speed is not None and 'advance_speed' in features:
        input_data['advance_speed'] = advance_speed

    # Create DataFrame
    X = pd.DataFrame([input_data])[features]

    # Predict
    prediction = model.predict(X)[0]

    # Display result
    print("\n" + "=" * 70)
    print("🎯 PREDICTION RESULT")
    print("=" * 70)

    print(f"\nInput Parameters:")
    print(f"   Geological Type:  {geological_type}")
    print(f"   Thrust Force:     {thrust_force} kN")
    print(f"   Torque:           {torque} kN·m")
    print(f"   RPM:              {rpm}")
    if earth_pressure:
        print(f"   Earth Pressure:   {earth_pressure} bar")
    if advance_speed:
        print(f"   Advance Speed:    {advance_speed} mm/min")

    print(f"\n📈 Predicted Advance Rate: {prediction:.2f} m/day")

    # Performance rating
    if prediction >= 35:
        rating = "🟢 Excellent - Above target performance"
        recommendation = "Maintain current parameters"
    elif prediction >= 25:
        rating = "🟢 Very Good - Target performance achieved"
        recommendation = "Continue with current settings"
    elif prediction >= 20:
        rating = "🟡 Good - Acceptable performance"
        recommendation = "Monitor and optimize if possible"
    elif prediction >= 15:
        rating = "🟠 Acceptable - Below target"
        recommendation = "Consider parameter adjustments"
    else:
        rating = "🔴 Poor - Significant improvement needed"
        recommendation = "Review all parameters and geology"

    print(f"\nPerformance Rating: {rating}")
    print(f"Recommendation: {recommendation}")

    # Uncertainty estimate
    model_rmse = model_package['performance']['rmse']
    print(f"\nEstimated Range: {prediction - model_rmse:.2f} to {prediction + model_rmse:.2f} m/day")
    print(f"(±{model_rmse:.2f} m/day based on model RMSE)")

    print("=" * 70)

    return prediction


def predict_batch(input_csv, output_csv='predictions.csv'):
    """
    Make predictions for multiple scenarios from CSV

    Args:
        input_csv: CSV file with input parameters
        output_csv: Where to save predictions

    Required columns in input CSV:
        - geological_type
        - thrust_force
        - torque
        - rpm
        Optional: earth_pressure, advance_speed
    """

    # Load model
    model_package = load_model()

    model = model_package['model']
    features = model_package['feature_columns']
    geo_mapping = model_package['geo_mapping']

    # Load input data
    print(f"\n📂 Loading scenarios from: {input_csv}")

    try:
        df = pd.read_csv(input_csv)
        print(f"✅ Loaded {len(df)} scenarios")
    except FileNotFoundError:
        print(f"\n❌ Error: File '{input_csv}' not found!")
        return None

    # Encode geological type
    if 'geological_type' in df.columns:
        df['geo_encoded'] = df['geological_type'].map(geo_mapping)

        # Check for unknown types
        unknown = df[df['geo_encoded'].isnull()]['geological_type'].unique()
        if len(unknown) > 0:
            print(f"\n⚠️ Warning: Unknown geological types found: {list(unknown)}")
            print("   These rows will be skipped.")
            df = df[df['geo_encoded'].notna()]

    # Check required features
    missing_features = [f for f in features if f not in df.columns]
    if missing_features:
        print(f"\n❌ Error: Missing required columns: {missing_features}")
        return None

    # Make predictions
    print("\n🔧 Generating predictions...")

    X = df[features]
    df['predicted_advance_rate'] = model.predict(X)

    # Add confidence intervals
    model_rmse = model_package['performance']['rmse']
    df['prediction_lower'] = df['predicted_advance_rate'] - model_rmse
    df['prediction_upper'] = df['predicted_advance_rate'] + model_rmse

    # Add performance ratings
    def get_rating(rate):
        if rate >= 35:
            return 'Excellent'
        elif rate >= 25:
            return 'Very Good'
        elif rate >= 20:
            return 'Good'
        elif rate >= 15:
            return 'Acceptable'
        else:
            return 'Poor'

    df['performance_rating'] = df['predicted_advance_rate'].apply(get_rating)

    # Save results
    df.to_csv(output_csv, index=False)

    print(f"✅ Predictions saved to: {output_csv}")

    # Summary statistics
    print("\n" + "=" * 70)
    print("📊 PREDICTION SUMMARY")
    print("=" * 70)

    print(f"\nTotal scenarios: {len(df)}")
    print(f"\nAdvance Rate Statistics:")
    print(f"   Mean:    {df['predicted_advance_rate'].mean():.2f} m/day")
    print(f"   Median:  {df['predicted_advance_rate'].median():.2f} m/day")
    print(f"   Min:     {df['predicted_advance_rate'].min():.2f} m/day")
    print(f"   Max:     {df['predicted_advance_rate'].max():.2f} m/day")
    print(f"   Std Dev: {df['predicted_advance_rate'].std():.2f} m/day")

    print(f"\nPerformance Rating Distribution:")
    for rating, count in df['performance_rating'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"   {rating:12s}: {count:4d} ({percentage:5.1f}%)")

    if 'geological_type' in df.columns:
        print(f"\nAverage Rate by Geological Type:")
        for geo_type, avg_rate in df.groupby('geological_type')['predicted_advance_rate'].mean().sort_values(ascending=False).items():
            print(f"   {geo_type:15s}: {avg_rate:.2f} m/day")

    print("=" * 70)

    return df


def interactive_prediction():
    """Interactive mode for making predictions"""

    print("\n" + "=" * 70)
    print("🎯 INTERACTIVE PREDICTION MODE")
    print("=" * 70)

    # Load model to get available options
    model_package = load_model()
    geo_types = list(model_package['geo_mapping'].keys())

    print("\nEnter your parameters:")

    # Get geological type
    print(f"\nAvailable geological types:")
    for i, geo in enumerate(geo_types, 1):
        print(f"   {i}. {geo}")

    while True:
        try:
            choice = int(input("\nSelect geological type (number): "))
            if 1 <= choice <= len(geo_types):
                geological_type = geo_types[choice - 1]
                break
            else:
                print(f"Please enter a number between 1 and {len(geo_types)}")
        except ValueError:
            print("Please enter a valid number")

    # Get other parameters
    thrust_force = float(input("Thrust Force (kN): "))
    torque = float(input("Torque (kN·m): "))
    rpm = float(input("RPM: "))

    # Optional parameters
    features = model_package['feature_columns']

    earth_pressure = None
    if 'earth_pressure' in features:
        ep_input = input("Earth Pressure (bar) [press Enter to skip]: ")
        if ep_input:
            earth_pressure = float(ep_input)

    advance_speed = None
    if 'advance_speed' in features:
        as_input = input("Advance Speed (mm/min) [press Enter to skip]: ")
        if as_input:
            advance_speed = float(as_input)

    # Make prediction
    predict_single(geological_type, thrust_force, torque, rpm,
                   earth_pressure, advance_speed)


def main():
    """Main function"""

    print("\n" + "=" * 70)
    print("🚀 MTBM ADVANCE RATE PREDICTION")
    print("=" * 70)

    print("\nChoose prediction mode:")
    print("   1. Single prediction (manual input)")
    print("   2. Batch predictions (from CSV)")
    print("   3. Interactive mode")
    print("   4. Example predictions")

    choice = input("\nEnter choice (1-4): ")

    if choice == '1':
        # Example single prediction
        print("\nExample: Predicting for Dense Sand conditions")
        predict_single(
            geological_type='dense_sand',
            thrust_force=1450,
            torque=245,
            rpm=8.5,
            earth_pressure=142
        )

    elif choice == '2':
        # Batch predictions
        input_file = input("\nEnter input CSV filename: ")
        output_file = input("Enter output CSV filename [predictions.csv]: ") or 'predictions.csv'
        predict_batch(input_file, output_file)

    elif choice == '3':
        # Interactive mode
        interactive_prediction()

    elif choice == '4':
        # Example predictions
        print("\n" + "=" * 70)
        print("📊 EXAMPLE PREDICTIONS")
        print("=" * 70)

        model_package = load_model()
        geo_types = list(model_package['geo_mapping'].keys())

        scenarios = [
            {
                'name': 'Optimal Parameters - Soft Clay',
                'geological_type': geo_types[0] if len(geo_types) > 0 else 'soft_clay',
                'thrust_force': 1300,
                'torque': 220,
                'rpm': 9.0,
                'earth_pressure': 130
            },
            {
                'name': 'Standard Parameters - Dense Sand',
                'geological_type': geo_types[1] if len(geo_types) > 1 else 'dense_sand',
                'thrust_force': 1500,
                'torque': 260,
                'rpm': 8.0,
                'earth_pressure': 145
            }
        ]

        for scenario in scenarios:
            name = scenario.pop('name')
            print(f"\n\nScenario: {name}")
            print("-" * 70)
            predict_single(**scenario)

    else:
        print("\nInvalid choice. Please run again and select 1-4.")


if __name__ == "__main__":
    main()
