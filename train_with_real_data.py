#!/usr/bin/env python3
"""
Train ML Model with Real MTBM Data
===================================

This script trains machine learning models using your real MTBM data.

Usage:
    python train_with_real_data.py

Requirements:
    - cleaned_mtbm_data.csv (created by load_real_data.py)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import sys
from datetime import datetime

def train_model(data_file='cleaned_mtbm_data.csv'):
    """Train ML model with real data"""

    print("=" * 70)
    print("🤖 TRAINING ML MODEL WITH REAL MTBM DATA")
    print("=" * 70)

    # Load data
    print(f"\n📂 Loading data from: {data_file}")

    try:
        df = pd.read_csv(data_file)
        print(f"✅ Loaded {len(df)} records")
    except FileNotFoundError:
        print(f"\n❌ Error: File '{data_file}' not found!")
        print("\nPlease run 'python load_real_data.py' first to prepare your data.")
        sys.exit(1)

    # Show data info
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print(f"Date range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")

    print(f"Columns: {list(df.columns)}")

    # Check geological types
    if 'geological_type' in df.columns:
        print(f"\n📊 Geological Types:")
        geo_counts = df['geological_type'].value_counts()
        for geo_type, count in geo_counts.items():
            print(f"   {geo_type:15s}: {count:4d} records")

        # Warning if too few records per type
        if geo_counts.min() < 30:
            print(f"\n⚠️ WARNING: Some geological types have < 30 records.")
            print("   Model accuracy may be lower for these types.")

    # Encode geological type
    print("\n🔧 Preparing features...")

    if 'geological_type' in df.columns:
        geo_types = df['geological_type'].unique()
        geo_mapping = {geo: idx for idx, geo in enumerate(geo_types)}
        df['geo_encoded'] = df['geological_type'].map(geo_mapping)
        print(f"✅ Encoded {len(geo_types)} geological types")
    else:
        print("⚠️ Warning: No geological_type column found")
        geo_mapping = {}

    # Select features
    possible_features = [
        'geo_encoded',
        'thrust_force',
        'torque',
        'rpm',
        'earth_pressure',
        'advance_speed'
    ]

    feature_columns = [col for col in possible_features if col in df.columns]

    print(f"\n📊 Features for training:")
    for i, feat in enumerate(feature_columns, 1):
        print(f"   {i}. {feat}")

    if len(feature_columns) < 3:
        print("\n❌ Error: Need at least 3 features for training")
        print("Please ensure your data has: geological_type, thrust_force, and advance_speed")
        sys.exit(1)

    # Check for target variable
    if 'advance_rate' not in df.columns:
        print("\n❌ Error: 'advance_rate' column not found!")
        if 'advance_speed' in df.columns:
            print("Calculating advance_rate from advance_speed...")
            df['advance_rate'] = df['advance_speed'] * 60 * 24 / 1000
        else:
            print("Cannot calculate advance_rate. Please check your data.")
            sys.exit(1)

    # Prepare data
    X = df[feature_columns].copy()
    y = df['advance_rate'].copy()

    # Remove NaN values
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    print(f"\n✅ Clean dataset ready: {len(X)} records")

    if len(X) < 100:
        print(f"\n⚠️ WARNING: Only {len(X)} records available for training.")
        print("   For best results, use 200+ records.")
        print("   Model accuracy may be limited.")

    # Split data
    test_size = 0.2 if len(X) > 50 else 0.3
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    print(f"\n📚 Data split:")
    print(f"   Training set: {len(X_train)} records ({(1-test_size)*100:.0f}%)")
    print(f"   Test set:     {len(X_test)} records ({test_size*100:.0f}%)")

    # Train multiple models
    print("\n" + "=" * 70)
    print("🔧 TRAINING MODELS")
    print("=" * 70)

    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            min_samples_split=5,
            random_state=42
        ),
        'Ridge Regression': Ridge(
            alpha=1.0,
            random_state=42
        )
    }

    results = {}
    best_score = -np.inf
    best_model = None
    best_model_name = None

    for name, model in models.items():
        print(f"\n{'=' * 70}")
        print(f"Training: {name}")
        print('=' * 70)

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        test_mae = mean_absolute_error(y_test, y_pred_test)

        # Cross-validation (if enough data)
        if len(X_train) >= 50:
            cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(X_train)//20), scoring='r2')
            cv_mean = cv_scores.mean()
            cv_std = cv_scores.std()
        else:
            cv_mean = None
            cv_std = None

        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': test_rmse,
            'mae': test_mae,
            'cv_mean': cv_mean,
            'cv_std': cv_std
        }

        # Display results
        print(f"\n📊 Performance Metrics:")
        print(f"   Training R²:     {train_r2:.3f}")
        print(f"   Test R²:         {test_r2:.3f}")
        print(f"   RMSE:            {test_rmse:.2f} m/day")
        print(f"   MAE:             {test_mae:.2f} m/day")

        if cv_mean is not None:
            print(f"   Cross-Val R²:    {cv_mean:.3f} (±{cv_std:.3f})")

        # Check for overfitting
        if train_r2 - test_r2 > 0.15:
            print(f"   ⚠️ Warning: Possible overfitting (train-test gap: {train_r2 - test_r2:.3f})")

        # Track best model
        if test_r2 > best_score:
            best_score = test_r2
            best_model = model
            best_model_name = name

    # Display best model
    print("\n" + "=" * 70)
    print("🏆 BEST MODEL")
    print("=" * 70)
    print(f"\nModel: {best_model_name}")
    print(f"Test R² Score: {results[best_model_name]['test_r2']:.3f}")
    print(f"RMSE: {results[best_model_name]['rmse']:.2f} m/day")
    print(f"MAE: {results[best_model_name]['mae']:.2f} m/day")

    # Performance interpretation
    test_r2 = results[best_model_name]['test_r2']
    if test_r2 >= 0.85:
        quality = "🟢 Excellent - Ready for production use"
    elif test_r2 >= 0.70:
        quality = "🟡 Good - Suitable for most applications"
    elif test_r2 >= 0.50:
        quality = "🟠 Acceptable - Use with caution"
    else:
        quality = "🔴 Poor - Need more/better data"

    print(f"\nModel Quality: {quality}")

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print("\n📊 Feature Importance:")
        importances = best_model.feature_importances_
        feature_imp = sorted(
            zip(feature_columns, importances),
            key=lambda x: x[1],
            reverse=True
        )

        for i, (feature, importance) in enumerate(feature_imp, 1):
            feature_display = feature.replace('geo_encoded', 'geological_type')
            bar = '█' * int(importance * 50)
            print(f"   {i}. {feature_display:20s} {bar} {importance:.3f}")

    # Save model
    model_filename = 'trained_advance_rate_model.pkl'

    model_package = {
        'model': best_model,
        'model_name': best_model_name,
        'feature_columns': feature_columns,
        'geo_mapping': geo_mapping,
        'performance': {
            'test_r2': results[best_model_name]['test_r2'],
            'rmse': results[best_model_name]['rmse'],
            'mae': results[best_model_name]['mae']
        },
        'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }

    joblib.dump(model_package, model_filename)
    print(f"\n💾 Model saved to: {model_filename}")

    # Save model info to text file
    info_filename = 'model_info.txt'
    with open(info_filename, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("MTBM ADVANCE RATE PREDICTION MODEL\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Training Date: {model_package['training_date']}\n")
        f.write(f"Model: {best_model_name}\n")
        f.write(f"Training Samples: {len(X_train)}\n")
        f.write(f"Test Samples: {len(X_test)}\n\n")

        f.write("Performance:\n")
        f.write(f"  Test R² Score: {results[best_model_name]['test_r2']:.3f}\n")
        f.write(f"  RMSE: {results[best_model_name]['rmse']:.2f} m/day\n")
        f.write(f"  MAE: {results[best_model_name]['mae']:.2f} m/day\n\n")

        f.write("Features Used:\n")
        for feat in feature_columns:
            f.write(f"  - {feat}\n")

        f.write("\nGeological Type Mapping:\n")
        for geo, code in geo_mapping.items():
            f.write(f"  {geo}: {code}\n")

        if hasattr(best_model, 'feature_importances_'):
            f.write("\nFeature Importance:\n")
            for feature, importance in feature_imp:
                f.write(f"  {feature}: {importance:.3f}\n")

    print(f"💾 Model info saved to: {info_filename}")

    # Sample predictions
    print("\n" + "=" * 70)
    print("🎯 SAMPLE PREDICTIONS")
    print("=" * 70)

    # Get sample from each geological type
    if 'geological_type' in df.columns:
        for geo_type in df['geological_type'].unique()[:3]:
            sample = df[df['geological_type'] == geo_type].iloc[0]

            sample_X = pd.DataFrame([{
                col: sample[col] if col != 'geo_encoded' else geo_mapping[geo_type]
                for col in feature_columns
            }])

            pred = best_model.predict(sample_X)[0]
            actual = sample['advance_rate']

            print(f"\n{geo_type}:")
            print(f"   Predicted: {pred:.2f} m/day")
            print(f"   Actual:    {actual:.2f} m/day")
            print(f"   Error:     {abs(pred - actual):.2f} m/day")

    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)
    print("\nNext Steps:")
    print("1. Review model performance above")
    print("2. If quality is good, use 'python make_predictions.py' to make predictions")
    print("3. If quality is poor, try collecting more data or adding features")
    print("\n" + "=" * 70)

    return best_model, feature_columns, geo_mapping


if __name__ == "__main__":
    train_model()
