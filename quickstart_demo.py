#!/usr/bin/env python3
"""
MTBM Machine Learning - Quick Start Demo
=========================================

This script demonstrates the complete ML workflow in 5 minutes.
Run this to see how the framework works with real examples.

Usage:
    python quickstart_demo.py
"""

import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys

print("=" * 60)
print("🚀 MTBM MACHINE LEARNING - QUICK START DEMO")
print("=" * 60)
print()

# Check if required packages are installed
try:
    import pandas as pd
    import numpy as np
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import r2_score, mean_squared_error
    print("✅ All required packages installed")
except ImportError as e:
    print(f"❌ Missing package: {e}")
    print("\nPlease install requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

print()
print("-" * 60)
print("STEP 1: Generate Synthetic MTBM Data")
print("-" * 60)

# Generate synthetic data
np.random.seed(42)
n_samples = 500

# Geological types
geo_types = ['soft_clay', 'dense_sand', 'hard_rock', 'mixed_ground']
geo_type_encoded = {
    'soft_clay': 0,
    'dense_sand': 1,
    'hard_rock': 2,
    'mixed_ground': 3
}

data = {
    'timestamp': pd.date_range(start='2024-01-01', periods=n_samples, freq='1H'),
    'chainage': np.cumsum(np.random.uniform(0.5, 2.0, n_samples)),
    'geological_type': np.random.choice(geo_types, n_samples),
    'thrust_force': np.random.uniform(800, 1800, n_samples),
    'torque': np.random.uniform(120, 400, n_samples),
    'rpm': np.random.uniform(5, 14, n_samples),
    'earth_pressure': np.random.uniform(100, 180, n_samples),
    'advance_speed': np.random.uniform(15, 45, n_samples)
}

df = pd.DataFrame(data)

# Add realistic relationships
for idx, row in df.iterrows():
    geo = row['geological_type']

    if geo == 'soft_clay':
        df.loc[idx, 'advance_speed'] = np.random.uniform(35, 45)
    elif geo == 'dense_sand':
        df.loc[idx, 'advance_speed'] = np.random.uniform(20, 30)
    elif geo == 'hard_rock':
        df.loc[idx, 'advance_speed'] = np.random.uniform(10, 20)
    else:  # mixed_ground
        df.loc[idx, 'advance_speed'] = np.random.uniform(15, 25)

# Add some noise
df['advance_speed'] += np.random.normal(0, 2, n_samples)

# Calculate advance rate (m/day)
df['advance_rate'] = df['advance_speed'] * 60 * 24 / 1000  # Convert mm/min to m/day

# Encode geological type
df['geo_encoded'] = df['geological_type'].map(geo_type_encoded)

print(f"✅ Generated {len(df)} MTBM operation records")
print(f"   Geological types: {df['geological_type'].nunique()}")
print(f"   Time span: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

print("Sample data:")
print(df[['timestamp', 'geological_type', 'thrust_force', 'advance_rate']].head())
print()

print("-" * 60)
print("STEP 2: Exploratory Data Analysis")
print("-" * 60)

print("\n📊 Ground Type Distribution:")
print(df['geological_type'].value_counts())

print("\n📈 Average Advance Rate by Geological Type:")
avg_by_geo = df.groupby('geological_type')['advance_rate'].mean().sort_values(ascending=False)
for geo, rate in avg_by_geo.items():
    print(f"   {geo:15s}: {rate:6.2f} m/day")

print("\n📉 Performance Statistics:")
print(f"   Overall average: {df['advance_rate'].mean():.2f} m/day")
print(f"   Best day: {df['advance_rate'].max():.2f} m/day")
print(f"   Worst day: {df['advance_rate'].min():.2f} m/day")
print(f"   Standard deviation: {df['advance_rate'].std():.2f} m/day")

print()
print("-" * 60)
print("STEP 3: Train Machine Learning Models")
print("-" * 60)

# Prepare features
feature_cols = ['geo_encoded', 'thrust_force', 'torque', 'rpm', 'earth_pressure']
X = df[feature_cols]
y = df['advance_rate']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"\n📚 Training set: {len(X_train)} samples")
print(f"📝 Test set: {len(X_test)} samples")

# Train models
models = {
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42, max_depth=5)
}

results = {}
print("\n🔧 Training models...")
for name, model in models.items():
    print(f"   Training {name}...", end=" ")
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    results[name] = {
        'model': model,
        'r2': r2,
        'rmse': rmse
    }
    print(f"✅ R² = {r2:.3f}, RMSE = {rmse:.2f}")

# Find best model
best_model_name = max(results, key=lambda x: results[x]['r2'])
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   R² Score: {results[best_model_name]['r2']:.3f}")
print(f"   RMSE: {results[best_model_name]['rmse']:.2f} m/day")

print()
print("-" * 60)
print("STEP 4: Feature Importance Analysis")
print("-" * 60)

if hasattr(best_model, 'feature_importances_'):
    importances = best_model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    print("\n📊 Top Features Affecting Advance Rate:")
    for idx, row in feature_importance.iterrows():
        feature_name = row['Feature'].replace('geo_encoded', 'geological_type')
        bar_length = int(row['Importance'] * 50)
        bar = '█' * bar_length
        print(f"   {feature_name:20s} {bar} {row['Importance']:.3f}")

print()
print("-" * 60)
print("STEP 5: Make Real-Time Predictions")
print("-" * 60)

# Test scenarios
scenarios = [
    {
        'name': 'Soft Clay - Optimal Parameters',
        'geo_encoded': geo_type_encoded['soft_clay'],
        'thrust_force': 1400,
        'torque': 220,
        'rpm': 9.0,
        'earth_pressure': 130
    },
    {
        'name': 'Hard Rock - Optimal Parameters',
        'geo_encoded': geo_type_encoded['hard_rock'],
        'thrust_force': 1750,
        'torque': 380,
        'rpm': 7.5,
        'earth_pressure': 165
    },
    {
        'name': 'Dense Sand - Low Thrust',
        'geo_encoded': geo_type_encoded['dense_sand'],
        'thrust_force': 900,
        'torque': 180,
        'rpm': 8.0,
        'earth_pressure': 120
    }
]

print("\n🎯 Testing Different Operating Scenarios:")
print()

for scenario in scenarios:
    name = scenario.pop('name')
    X_scenario = pd.DataFrame([scenario])
    prediction = best_model.predict(X_scenario)[0]

    print(f"Scenario: {name}")
    print(f"   Predicted Advance Rate: {prediction:.2f} m/day")

    # Performance rating
    if prediction >= 30:
        rating = "🟢 Excellent"
    elif prediction >= 20:
        rating = "🟡 Good"
    elif prediction >= 15:
        rating = "🟠 Acceptable"
    else:
        rating = "🔴 Poor"

    print(f"   Performance Rating: {rating}")
    print()

print("-" * 60)
print("STEP 6: Optimization Recommendation")
print("-" * 60)

# Find optimal parameters for dense sand
print("\n💡 Finding Optimal Parameters for Dense Sand...")

best_rate = 0
best_params = None

# Grid search (simplified)
for thrust in [1200, 1400, 1600]:
    for rpm in [7.0, 8.0, 9.0]:
        for pressure in [130, 145, 160]:
            test_params = pd.DataFrame([{
                'geo_encoded': geo_type_encoded['dense_sand'],
                'thrust_force': thrust,
                'torque': 250,  # Fixed
                'rpm': rpm,
                'earth_pressure': pressure
            }])

            pred_rate = best_model.predict(test_params)[0]

            if pred_rate > best_rate:
                best_rate = pred_rate
                best_params = {
                    'thrust_force': thrust,
                    'rpm': rpm,
                    'earth_pressure': pressure
                }

print("\n✅ Optimal Parameters Found:")
print(f"   Thrust Force: {best_params['thrust_force']} kN")
print(f"   RPM: {best_params['rpm']}")
print(f"   Earth Pressure: {best_params['earth_pressure']} bar")
print(f"\n📈 Expected Advance Rate: {best_rate:.2f} m/day")

# Calculate improvement
baseline_rate = df[df['geological_type'] == 'dense_sand']['advance_rate'].mean()
improvement = ((best_rate / baseline_rate) - 1) * 100
print(f"🚀 Improvement over baseline: +{improvement:.1f}%")

print()
print("=" * 60)
print("✅ DEMO COMPLETE!")
print("=" * 60)
print()
print("📚 What you learned:")
print("   1. How to generate and load MTBM data")
print("   2. Exploratory data analysis techniques")
print("   3. Training multiple ML models")
print("   4. Feature importance analysis")
print("   5. Making real-time predictions")
print("   6. Parameter optimization")
print()
print("📖 Next Steps:")
print("   • Read PRACTICAL_GUIDE.md for detailed tutorials")
print("   • Explore unified_mtbm_ml_framework.py for advanced features")
print("   • Try avn2400_advanced_measurement_ml.py for precision analysis")
print("   • Check legacy/ folder for specialized tools")
print()
print("💡 Pro Tip: Replace synthetic data with your real MTBM data")
print("   for production-ready predictions!")
print()
print("=" * 60)
