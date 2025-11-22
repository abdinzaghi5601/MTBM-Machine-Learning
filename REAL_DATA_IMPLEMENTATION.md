# Using Real MTBM Data - Complete Implementation Guide

## 🎯 Overview
This guide shows you **exactly** how to replace synthetic data with your real MTBM data and get predictions.

---

## 📊 Step 1: Understand Your Data Format

### What data do you need?

**Minimum Required Columns:**
- `timestamp` or `date` - When the measurement was taken
- `geological_type` or `ground_type` - Type of ground (clay, sand, rock, etc.)
- `thrust_force` or `total_thrust` - Thrust force in kN
- `advance_speed` or `penetration_rate` - Speed in mm/min
- `torque` - Cutting torque in kN·m
- `rpm` - Rotation speed

**Optional but Helpful:**
- `chainage` - Position in tunnel (meters)
- `earth_pressure` - Pressure in bar
- `cutter_wear` - Wear measurements
- `deviation_horizontal` - Steering deviation
- `deviation_vertical` - Vertical deviation

### Common Data Sources

**Source 1: Excel Files (.xlsx, .xls)**
```
MTBM_Data_2024.xlsx
├── Sheet1: Daily Operations
├── Sheet2: Geological Log
└── Sheet3: Equipment Status
```

**Source 2: CSV Files**
```
mtbm_operations_jan.csv
mtbm_operations_feb.csv
mtbm_operations_mar.csv
```

**Source 3: Database**
```
SQL Server / MySQL / PostgreSQL
Table: mtbm_operations
```

**Source 4: SCADA/PLC Exports**
```
Raw sensor logs (often CSV or TXT format)
```

---

## 📝 Step 2: Prepare Your Data

### Option A: Excel File

Create this script: **`load_excel_data.py`**

```python
import pandas as pd
import numpy as np

def load_mtbm_excel(filepath, sheet_name='Sheet1'):
    """
    Load MTBM data from Excel file

    Args:
        filepath: Path to your Excel file
        sheet_name: Name of the sheet with data

    Returns:
        Clean pandas DataFrame ready for ML
    """

    print(f"📂 Loading data from: {filepath}")

    # Load Excel
    df = pd.read_excel(filepath, sheet_name=sheet_name)

    print(f"✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"\nOriginal columns: {list(df.columns)}")

    # Rename columns to standard names
    column_mapping = {
        # Add your actual column names here
        'Date': 'timestamp',
        'Ground Type': 'geological_type',
        'Thrust (kN)': 'thrust_force',
        'Speed (mm/min)': 'advance_speed',
        'Torque (kNm)': 'torque',
        'RPM': 'rpm',
        'Pressure (bar)': 'earth_pressure',
        'Chainage (m)': 'chainage'
    }

    # Rename columns
    df = df.rename(columns=column_mapping)

    print(f"\n✅ Renamed columns to standard format")
    print(f"New columns: {list(df.columns)}")

    # Clean data
    df = clean_mtbm_data(df)

    return df


def clean_mtbm_data(df):
    """
    Clean and prepare MTBM data for ML
    """

    print("\n🧹 Cleaning data...")

    # 1. Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        print("✅ Converted timestamp to datetime")

    # 2. Remove completely empty rows
    df = df.dropna(how='all')
    print(f"✅ Removed empty rows. Remaining: {len(df)}")

    # 3. Handle missing values
    # Option A: Remove rows with missing critical values
    critical_cols = ['geological_type', 'thrust_force', 'advance_speed']
    df = df.dropna(subset=critical_cols)
    print(f"✅ Removed rows with missing critical data. Remaining: {len(df)}")

    # Option B: Fill missing values with mean (for numerical columns)
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        if df[col].isnull().any():
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"   Filled {col} missing values with mean")

    # 4. Remove outliers (values that are clearly errors)
    # Example: thrust force should be between 0 and 3000 kN
    if 'thrust_force' in df.columns:
        original_len = len(df)
        df = df[(df['thrust_force'] > 0) & (df['thrust_force'] < 3000)]
        removed = original_len - len(df)
        if removed > 0:
            print(f"✅ Removed {removed} outlier rows (thrust_force)")

    # 5. Standardize geological types
    if 'geological_type' in df.columns:
        # Map your geological names to standard categories
        geo_mapping = {
            'Clay': 'soft_clay',
            'CLAY': 'soft_clay',
            'Soft Clay': 'soft_clay',
            'Sand': 'dense_sand',
            'SAND': 'dense_sand',
            'Dense Sand': 'dense_sand',
            'Rock': 'hard_rock',
            'ROCK': 'hard_rock',
            'Hard Rock': 'hard_rock',
            'Mixed': 'mixed_ground',
            'MIXED': 'mixed_ground'
        }

        df['geological_type'] = df['geological_type'].replace(geo_mapping)
        print("✅ Standardized geological type names")
        print(f"   Categories: {df['geological_type'].unique()}")

    # 6. Calculate advance rate if not present
    if 'advance_speed' in df.columns and 'advance_rate' not in df.columns:
        # Convert mm/min to m/day
        df['advance_rate'] = df['advance_speed'] * 60 * 24 / 1000
        print("✅ Calculated advance_rate (m/day) from advance_speed")

    # 7. Sort by timestamp
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        print("✅ Sorted by timestamp")

    print(f"\n✅ Data cleaning complete!")
    print(f"Final dataset: {len(df)} rows × {len(df.columns)} columns")

    return df


# Example usage
if __name__ == "__main__":
    # Replace with your actual file path
    df = load_mtbm_excel('MTBM_Data_2024.xlsx', sheet_name='Operations')

    # Save cleaned data
    df.to_csv('cleaned_mtbm_data.csv', index=False)
    print("\n💾 Saved cleaned data to: cleaned_mtbm_data.csv")

    # Show summary
    print("\n📊 Data Summary:")
    print(df.describe())

    print("\n📈 Ground Type Distribution:")
    print(df['geological_type'].value_counts())
```

**How to use this:**

1. **Save the script above** as `load_excel_data.py`

2. **Edit the column mapping** to match YOUR Excel columns:
```python
column_mapping = {
    'Your Column Name': 'standard_name',
    'Date/Time': 'timestamp',
    'Soil Type': 'geological_type',
    # ... add all your columns
}
```

3. **Run the script:**
```bash
python load_excel_data.py
```

4. **You'll get** a clean CSV file: `cleaned_mtbm_data.csv`

---

### Option B: CSV File

Create this script: **`load_csv_data.py`**

```python
import pandas as pd

def load_mtbm_csv(filepath):
    """Load MTBM data from CSV file"""

    print(f"📂 Loading CSV: {filepath}")

    # Try different separators (comma, semicolon, tab)
    separators = [',', ';', '\t']

    for sep in separators:
        try:
            df = pd.read_csv(filepath, sep=sep)
            if len(df.columns) > 1:  # Successfully parsed
                print(f"✅ Loaded with separator: '{sep}'")
                break
        except:
            continue

    print(f"Loaded {len(df)} rows")
    print(f"Columns: {list(df.columns)}")

    # Use the same cleaning function from above
    df = clean_mtbm_data(df)

    return df


# Example usage
if __name__ == "__main__":
    df = load_mtbm_csv('your_mtbm_data.csv')
    df.to_csv('cleaned_mtbm_data.csv', index=False)
```

---

### Option C: Multiple CSV Files

```python
import pandas as pd
import glob

def load_multiple_csv_files(folder_path):
    """Load and combine multiple CSV files"""

    print(f"📂 Loading all CSV files from: {folder_path}")

    # Find all CSV files
    csv_files = glob.glob(f"{folder_path}/*.csv")

    print(f"Found {len(csv_files)} CSV files")

    # Load and combine all files
    dfs = []
    for file in csv_files:
        print(f"  Loading: {file}")
        df = pd.read_csv(file)
        dfs.append(df)

    # Combine all dataframes
    combined_df = pd.concat(dfs, ignore_index=True)

    print(f"\n✅ Combined {len(csv_files)} files")
    print(f"Total records: {len(combined_df)}")

    # Clean combined data
    combined_df = clean_mtbm_data(combined_df)

    return combined_df


# Example usage
if __name__ == "__main__":
    df = load_multiple_csv_files('mtbm_data_folder')
    df.to_csv('all_mtbm_data.csv', index=False)
```

---

### Option D: Database Connection

```python
import pandas as pd
from sqlalchemy import create_engine

def load_from_database(connection_string, query=None, table_name=None):
    """
    Load MTBM data from database

    Args:
        connection_string: Database connection string
        query: SQL query (optional)
        table_name: Table name (optional)
    """

    print("🔌 Connecting to database...")

    # Create connection
    engine = create_engine(connection_string)

    # Load data
    if query:
        print(f"📊 Executing query...")
        df = pd.read_sql(query, engine)
    elif table_name:
        print(f"📊 Loading table: {table_name}")
        df = pd.read_sql_table(table_name, engine)
    else:
        raise ValueError("Provide either query or table_name")

    print(f"✅ Loaded {len(df)} records from database")

    # Clean data
    df = clean_mtbm_data(df)

    return df


# Example usage for different databases:

# SQL Server
connection_string = "mssql+pyodbc://username:password@server/database?driver=ODBC+Driver+17+for+SQL+Server"

# MySQL
connection_string = "mysql+pymysql://username:password@localhost/database"

# PostgreSQL
connection_string = "postgresql://username:password@localhost/database"

# Load data
df = load_from_database(
    connection_string,
    query="SELECT * FROM mtbm_operations WHERE date >= '2024-01-01'"
)
```

---

## 🤖 Step 3: Train Models with Real Data

Create this script: **`train_with_real_data.py`**

```python
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib  # For saving models

def train_advance_rate_model(data_file):
    """
    Train ML model to predict advance rate using real data

    Args:
        data_file: Path to cleaned CSV file
    """

    print("=" * 60)
    print("🤖 TRAINING ML MODEL WITH REAL DATA")
    print("=" * 60)

    # Load data
    print(f"\n📂 Loading data from: {data_file}")
    df = pd.read_csv(data_file)

    print(f"✅ Loaded {len(df)} records")
    print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Encode geological type
    print("\n🔧 Encoding geological types...")
    geo_types = df['geological_type'].unique()
    geo_mapping = {geo: idx for idx, geo in enumerate(geo_types)}
    df['geo_encoded'] = df['geological_type'].map(geo_mapping)

    print(f"Geological types: {list(geo_types)}")

    # Select features
    feature_columns = []

    # Add available features
    possible_features = [
        'geo_encoded',
        'thrust_force',
        'torque',
        'rpm',
        'earth_pressure',
        'advance_speed'
    ]

    for col in possible_features:
        if col in df.columns:
            feature_columns.append(col)

    print(f"\n📊 Features for training: {feature_columns}")

    # Prepare data
    X = df[feature_columns]
    y = df['advance_rate']

    # Remove any remaining NaN
    mask = ~(X.isnull().any(axis=1) | y.isnull())
    X = X[mask]
    y = y[mask]

    print(f"✅ Clean dataset: {len(X)} records")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"\n📚 Training set: {len(X_train)} records")
    print(f"📝 Test set: {len(X_test)} records")

    # Train models
    print("\n🔧 Training models...")

    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            random_state=42,
            n_jobs=-1
        ),
        'Gradient Boosting': GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
    }

    results = {}

    for name, model in models.items():
        print(f"\n  Training {name}...")

        # Train
        model.fit(X_train, y_train)

        # Predict
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        # Evaluate
        train_r2 = r2_score(y_train, y_pred_train)
        test_r2 = r2_score(y_test, y_pred_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))

        results[name] = {
            'model': model,
            'train_r2': train_r2,
            'test_r2': test_r2,
            'rmse': test_rmse
        }

        print(f"  ✅ Train R²: {train_r2:.3f}")
        print(f"  ✅ Test R²: {test_r2:.3f}")
        print(f"  ✅ RMSE: {test_rmse:.2f} m/day")

    # Find best model
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']

    print("\n" + "=" * 60)
    print(f"🏆 BEST MODEL: {best_model_name}")
    print("=" * 60)
    print(f"Test R² Score: {results[best_model_name]['test_r2']:.3f}")
    print(f"RMSE: {results[best_model_name]['rmse']:.2f} m/day")

    # Feature importance
    if hasattr(best_model, 'feature_importances_'):
        print("\n📊 Feature Importance:")
        importances = best_model.feature_importances_
        for feature, importance in sorted(
            zip(feature_columns, importances),
            key=lambda x: x[1],
            reverse=True
        ):
            print(f"  {feature:20s}: {importance:.3f}")

    # Save model
    model_filename = 'trained_advance_rate_model.pkl'
    joblib.dump({
        'model': best_model,
        'feature_columns': feature_columns,
        'geo_mapping': geo_mapping
    }, model_filename)

    print(f"\n💾 Model saved to: {model_filename}")

    # Save feature info
    with open('model_info.txt', 'w') as f:
        f.write(f"Model: {best_model_name}\n")
        f.write(f"Training Date: {pd.Timestamp.now()}\n")
        f.write(f"Features: {feature_columns}\n")
        f.write(f"Geological Mapping: {geo_mapping}\n")
        f.write(f"Test R²: {results[best_model_name]['test_r2']:.3f}\n")
        f.write(f"RMSE: {results[best_model_name]['rmse']:.2f} m/day\n")

    print("💾 Model info saved to: model_info.txt")

    return best_model, feature_columns, geo_mapping


if __name__ == "__main__":
    # Train with your cleaned data
    model, features, geo_map = train_advance_rate_model('cleaned_mtbm_data.csv')

    print("\n✅ Training complete!")
    print("\nYou can now use this model to make predictions!")
```

**Run this:**
```bash
python train_with_real_data.py
```

---

## 🎯 Step 4: Make Predictions with Your Trained Model

Create this script: **`make_predictions.py`**

```python
import pandas as pd
import joblib

def load_trained_model(model_file='trained_advance_rate_model.pkl'):
    """Load the trained model"""

    print(f"📂 Loading model from: {model_file}")

    model_data = joblib.load(model_file)

    model = model_data['model']
    features = model_data['feature_columns']
    geo_mapping = model_data['geo_mapping']

    print("✅ Model loaded successfully")
    print(f"Features: {features}")
    print(f"Geological types: {list(geo_mapping.keys())}")

    return model, features, geo_mapping


def predict_advance_rate(geological_type, thrust_force, torque, rpm,
                         earth_pressure=None, advance_speed=None):
    """
    Predict advance rate for given conditions

    Args:
        geological_type: Type of ground (e.g., 'soft_clay', 'dense_sand')
        thrust_force: Thrust force in kN
        torque: Torque in kN·m
        rpm: Rotation speed
        earth_pressure: Earth pressure in bar (optional)
        advance_speed: Current advance speed mm/min (optional)

    Returns:
        Predicted advance rate in m/day
    """

    # Load model
    model, features, geo_mapping = load_trained_model()

    # Encode geological type
    if geological_type not in geo_mapping:
        print(f"⚠️ Unknown geological type: {geological_type}")
        print(f"Available types: {list(geo_mapping.keys())}")
        return None

    geo_encoded = geo_mapping[geological_type]

    # Prepare input data
    input_data = {
        'geo_encoded': geo_encoded,
        'thrust_force': thrust_force,
        'torque': torque,
        'rpm': rpm
    }

    # Add optional features if provided and used in model
    if earth_pressure is not None and 'earth_pressure' in features:
        input_data['earth_pressure'] = earth_pressure

    if advance_speed is not None and 'advance_speed' in features:
        input_data['advance_speed'] = advance_speed

    # Create DataFrame with correct feature order
    X = pd.DataFrame([input_data])[features]

    # Predict
    prediction = model.predict(X)[0]

    print("\n" + "=" * 60)
    print("🎯 PREDICTION RESULT")
    print("=" * 60)
    print(f"Geological Type: {geological_type}")
    print(f"Thrust Force: {thrust_force} kN")
    print(f"Torque: {torque} kN·m")
    print(f"RPM: {rpm}")
    if earth_pressure:
        print(f"Earth Pressure: {earth_pressure} bar")

    print(f"\n📈 Predicted Advance Rate: {prediction:.2f} m/day")

    # Performance rating
    if prediction >= 30:
        rating = "🟢 Excellent"
    elif prediction >= 20:
        rating = "🟡 Good"
    elif prediction >= 15:
        rating = "🟠 Acceptable"
    else:
        rating = "🔴 Poor - Consider adjusting parameters"

    print(f"Performance Rating: {rating}")
    print("=" * 60)

    return prediction


def predict_from_csv(input_csv, output_csv='predictions.csv'):
    """
    Make predictions for multiple records from CSV

    Args:
        input_csv: CSV file with columns matching model features
        output_csv: Where to save predictions
    """

    print(f"\n📂 Loading data from: {input_csv}")

    # Load data
    df = pd.read_csv(input_csv)

    print(f"✅ Loaded {len(df)} records for prediction")

    # Load model
    model, features, geo_mapping = load_trained_model()

    # Encode geological type if present
    if 'geological_type' in df.columns:
        df['geo_encoded'] = df['geological_type'].map(geo_mapping)

    # Make predictions
    X = df[features]
    df['predicted_advance_rate'] = model.predict(X)

    # Save results
    df.to_csv(output_csv, index=False)

    print(f"✅ Predictions saved to: {output_csv}")
    print(f"\n📊 Prediction Statistics:")
    print(df['predicted_advance_rate'].describe())

    return df


# Example usage
if __name__ == "__main__":

    print("=" * 60)
    print("🎯 MTBM ADVANCE RATE PREDICTION")
    print("=" * 60)

    # Example 1: Single prediction
    print("\n--- Example 1: Single Prediction ---")

    prediction = predict_advance_rate(
        geological_type='dense_sand',
        thrust_force=1450,
        torque=245,
        rpm=8.5,
        earth_pressure=142
    )

    # Example 2: Batch predictions from CSV
    print("\n\n--- Example 2: Batch Predictions ---")

    # Create sample input file
    sample_data = pd.DataFrame({
        'geological_type': ['soft_clay', 'dense_sand', 'hard_rock'],
        'thrust_force': [1200, 1450, 1750],
        'torque': [200, 245, 350],
        'rpm': [9.0, 8.5, 7.5],
        'earth_pressure': [130, 142, 160]
    })

    sample_data.to_csv('sample_input.csv', index=False)

    # Make predictions
    results = predict_from_csv('sample_input.csv', 'sample_predictions.csv')

    print("\nPredictions:")
    print(results[['geological_type', 'thrust_force', 'predicted_advance_rate']])
```

**Run this:**
```bash
python make_predictions.py
```

---

## 📋 Step 5: Complete Workflow Example

Here's a complete workflow from start to finish:

```bash
# 1. Prepare your data
python load_excel_data.py
# Output: cleaned_mtbm_data.csv

# 2. Train model
python train_with_real_data.py
# Output: trained_advance_rate_model.pkl

# 3. Make predictions
python make_predictions.py
# Output: Predictions displayed and saved
```

---

## 🎓 Real Example Walkthrough

Let's say you have this Excel file: `MTBM_Project_2024.xlsx`

### Your Excel looks like this:
```
| Date       | Chainage | Ground  | Thrust | Speed  | Torque | RPM |
|------------|----------|---------|--------|--------|--------|-----|
| 2024-01-01 | 10.5     | Clay    | 1250   | 35.2   | 210    | 8.5 |
| 2024-01-01 | 11.2     | Clay    | 1280   | 36.1   | 215    | 8.7 |
| 2024-01-02 | 12.8     | Sand    | 1450   | 28.3   | 245    | 8.2 |
```

### Step 1: Edit `load_excel_data.py`
```python
column_mapping = {
    'Date': 'timestamp',
    'Ground': 'geological_type',
    'Thrust': 'thrust_force',
    'Speed': 'advance_speed',
    'Torque': 'torque',
    'RPM': 'rpm'
}
```

### Step 2: Run
```bash
python load_excel_data.py
```

### Step 3: Check output
```
✅ Loaded 1547 rows
✅ Cleaned data: 1542 rows
💾 Saved: cleaned_mtbm_data.csv
```

### Step 4: Train model
```bash
python train_with_real_data.py
```

### Step 5: Predict
```python
from make_predictions import predict_advance_rate

# Predict for tomorrow
prediction = predict_advance_rate(
    geological_type='soft_clay',
    thrust_force=1300,
    torque=220,
    rpm=8.5
)

# Result: Predicted Advance Rate: 42.3 m/day
```

---

## 🚨 Troubleshooting Common Issues

### Issue 1: "Column not found"
**Problem:** Your columns have different names

**Solution:** Update the `column_mapping` dictionary:
```python
column_mapping = {
    'YOUR_ACTUAL_COLUMN_NAME': 'standard_name'
}
```

### Issue 2: "Model performance is poor (R² < 0.5)"
**Possible reasons:**
1. Not enough data (need 100+ records minimum)
2. Missing important features
3. Data quality issues

**Solution:**
```python
# Check data quality
print(df.describe())
print(df['geological_type'].value_counts())

# Need at least 30 records per geological type
```

### Issue 3: "Prediction seems wrong"
**Solution:** Check input units:
```python
# Make sure units match:
thrust_force = 1450  # kN (not tons)
torque = 245  # kN·m
rpm = 8.5  # revolutions per minute
advance_speed = 35  # mm/min (not m/day)
```

---

## ✅ Summary Checklist

- [ ] Prepared real data file (Excel/CSV/Database)
- [ ] Created and ran `load_excel_data.py` or `load_csv_data.py`
- [ ] Got `cleaned_mtbm_data.csv` with correct columns
- [ ] Trained model with `train_with_real_data.py`
- [ ] Got `trained_advance_rate_model.pkl` file
- [ ] Made test prediction with `make_predictions.py`
- [ ] Verified prediction makes sense
- [ ] Ready to use in production!

---

## 🎯 Next Steps

1. **Integrate with existing frameworks:**
```python
# Use with unified framework
from unified_mtbm_ml_framework import UnifiedMTBMFramework

framework = UnifiedMTBMFramework()
framework.load_trained_model('trained_advance_rate_model.pkl')
framework.predict(your_data)
```

2. **Set up automated monitoring**
3. **Create Power BI dashboard with real data**
4. **Implement real-time predictions**

---

**You now have everything you need to use REAL DATA! 🚀**

Start with your Excel or CSV file and follow the steps above.
