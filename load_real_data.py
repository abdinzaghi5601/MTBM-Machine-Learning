#!/usr/bin/env python3
"""
Load and Clean Real MTBM Data
==============================

This script helps you load your real MTBM data from Excel/CSV files
and prepares it for machine learning.

Usage:
    python load_real_data.py

Then edit the file paths and column names to match your data.
"""

import pandas as pd
import numpy as np
import sys
from datetime import datetime

def load_mtbm_excel(filepath, sheet_name='Sheet1'):
    """
    Load MTBM data from Excel file

    Args:
        filepath: Path to your Excel file
        sheet_name: Name of the sheet with data

    Returns:
        Clean pandas DataFrame ready for ML
    """

    print("=" * 60)
    print("📂 LOADING MTBM DATA FROM EXCEL")
    print("=" * 60)
    print(f"\nFile: {filepath}")
    print(f"Sheet: {sheet_name}")

    try:
        # Load Excel
        df = pd.read_excel(filepath, sheet_name=sheet_name)
        print(f"\n✅ Successfully loaded {len(df)} rows, {len(df.columns)} columns")
    except FileNotFoundError:
        print(f"\n❌ Error: File not found: {filepath}")
        print("Please check the file path and try again.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error loading file: {e}")
        sys.exit(1)

    print(f"\nOriginal columns:")
    for i, col in enumerate(df.columns, 1):
        print(f"  {i}. {col}")

    # IMPORTANT: Edit this mapping to match YOUR Excel columns!
    # Format: 'Your Column Name': 'standard_name'
    column_mapping = {
        # Date/Time
        'Date': 'timestamp',
        'DateTime': 'timestamp',
        'Time': 'timestamp',

        # Position
        'Chainage': 'chainage',
        'Position': 'chainage',
        'Distance': 'chainage',

        # Geological
        'Ground Type': 'geological_type',
        'Soil Type': 'geological_type',
        'Geology': 'geological_type',
        'Ground': 'geological_type',

        # Machine Parameters
        'Thrust': 'thrust_force',
        'Thrust Force': 'thrust_force',
        'Total Thrust': 'thrust_force',
        'Thrust (kN)': 'thrust_force',

        'Torque': 'torque',
        'Cutting Torque': 'torque',
        'Torque (kNm)': 'torque',

        'RPM': 'rpm',
        'Revolution': 'rpm',
        'Rotation Speed': 'rpm',

        'Speed': 'advance_speed',
        'Advance Speed': 'advance_speed',
        'Penetration Rate': 'advance_speed',
        'Speed (mm/min)': 'advance_speed',

        'Pressure': 'earth_pressure',
        'Earth Pressure': 'earth_pressure',
        'Face Pressure': 'earth_pressure',
        'Pressure (bar)': 'earth_pressure',

        # Deviations (optional)
        'Horizontal Deviation': 'deviation_horizontal',
        'Vertical Deviation': 'deviation_vertical',
        'Total Deviation': 'total_deviation'
    }

    # Rename columns
    df = df.rename(columns=column_mapping)

    print(f"\n✅ Columns renamed to standard format")
    print(f"\nStandardized columns found:")
    standard_cols = [col for col in df.columns if col in column_mapping.values()]
    for col in standard_cols:
        print(f"  ✓ {col}")

    # Check for required columns
    required = ['geological_type', 'thrust_force', 'advance_speed']
    missing = [col for col in required if col not in df.columns]

    if missing:
        print(f"\n⚠️ WARNING: Missing required columns: {missing}")
        print("\nPlease update the column_mapping dictionary in this script")
        print("to match your Excel column names.")
        return None

    # Clean data
    df = clean_mtbm_data(df)

    return df


def load_mtbm_csv(filepath):
    """Load MTBM data from CSV file"""

    print("=" * 60)
    print("📂 LOADING MTBM DATA FROM CSV")
    print("=" * 60)
    print(f"\nFile: {filepath}")

    # Try different separators
    separators = [',', ';', '\t', '|']

    df = None
    for sep in separators:
        try:
            test_df = pd.read_csv(filepath, sep=sep, nrows=5)
            if len(test_df.columns) > 1:
                df = pd.read_csv(filepath, sep=sep)
                print(f"✅ Loaded with separator: '{sep}'")
                print(f"Rows: {len(df)}, Columns: {len(df.columns)}")
                break
        except:
            continue

    if df is None:
        print("❌ Could not determine CSV separator")
        return None

    # Use same column mapping as Excel
    # You can edit this section to match your CSV columns
    column_mapping = {
        'Date': 'timestamp',
        'Ground Type': 'geological_type',
        'Thrust': 'thrust_force',
        'Speed': 'advance_speed',
        'Torque': 'torque',
        'RPM': 'rpm',
        'Pressure': 'earth_pressure'
    }

    df = df.rename(columns=column_mapping)
    df = clean_mtbm_data(df)

    return df


def clean_mtbm_data(df):
    """
    Clean and prepare MTBM data for ML
    """

    print("\n" + "=" * 60)
    print("🧹 CLEANING DATA")
    print("=" * 60)

    original_rows = len(df)

    # 1. Convert timestamp to datetime
    if 'timestamp' in df.columns:
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            print("✅ Converted timestamp to datetime")
            print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        except:
            print("⚠️ Could not convert timestamp column")

    # 2. Remove completely empty rows
    df = df.dropna(how='all')
    print(f"✅ Removed empty rows. Remaining: {len(df)}")

    # 3. Handle missing values in critical columns
    critical_cols = ['geological_type', 'thrust_force', 'advance_speed']
    available_critical = [col for col in critical_cols if col in df.columns]

    if available_critical:
        before = len(df)
        df = df.dropna(subset=available_critical)
        removed = before - len(df)
        if removed > 0:
            print(f"✅ Removed {removed} rows with missing critical data")

    # 4. Fill missing values in non-critical numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    for col in numerical_cols:
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            df[col].fillna(df[col].mean(), inplace=True)
            print(f"   Filled {missing_count} missing values in {col} with mean")

    # 5. Remove outliers
    if 'thrust_force' in df.columns:
        mask = (df['thrust_force'] > 0) & (df['thrust_force'] < 3000)
        removed = len(df) - mask.sum()
        df = df[mask]
        if removed > 0:
            print(f"✅ Removed {removed} outlier rows (thrust_force)")

    if 'torque' in df.columns:
        mask = (df['torque'] > 0) & (df['torque'] < 1000)
        removed = len(df) - mask.sum()
        df = df[mask]
        if removed > 0:
            print(f"✅ Removed {removed} outlier rows (torque)")

    if 'rpm' in df.columns:
        mask = (df['rpm'] > 0) & (df['rpm'] < 20)
        removed = len(df) - mask.sum()
        df = df[mask]
        if removed > 0:
            print(f"✅ Removed {removed} outlier rows (rpm)")

    # 6. Standardize geological types
    if 'geological_type' in df.columns:
        # Convert to lowercase and strip whitespace
        df['geological_type'] = df['geological_type'].astype(str).str.lower().str.strip()

        # Map variations to standard names
        geo_mapping = {
            'clay': 'soft_clay',
            'soft clay': 'soft_clay',
            'softclay': 'soft_clay',
            'sand': 'dense_sand',
            'dense sand': 'dense_sand',
            'densesand': 'dense_sand',
            'rock': 'hard_rock',
            'hard rock': 'hard_rock',
            'hardrock': 'hard_rock',
            'mixed': 'mixed_ground',
            'mixed ground': 'mixed_ground',
            'mixedground': 'mixed_ground'
        }

        df['geological_type'] = df['geological_type'].replace(geo_mapping)

        print("✅ Standardized geological type names")
        print(f"\n   Ground type distribution:")
        for geo_type, count in df['geological_type'].value_counts().items():
            percentage = (count / len(df)) * 100
            print(f"      {geo_type:15s}: {count:4d} ({percentage:5.1f}%)")

    # 7. Calculate advance rate if not present
    if 'advance_speed' in df.columns and 'advance_rate' not in df.columns:
        # Convert mm/min to m/day
        df['advance_rate'] = df['advance_speed'] * 60 * 24 / 1000
        print("✅ Calculated advance_rate (m/day) from advance_speed")

    # 8. Sort by timestamp if available
    if 'timestamp' in df.columns:
        df = df.sort_values('timestamp')
        df = df.reset_index(drop=True)
        print("✅ Sorted by timestamp")

    # 9. Final summary
    print("\n" + "=" * 60)
    print("✅ DATA CLEANING COMPLETE")
    print("=" * 60)
    print(f"Original rows: {original_rows}")
    print(f"Final rows: {len(df)}")
    print(f"Rows removed: {original_rows - len(df)} ({((original_rows - len(df)) / original_rows * 100):.1f}%)")
    print(f"Final columns: {len(df.columns)}")

    return df


def main():
    """Main function"""

    print("\n" + "=" * 60)
    print("🚀 REAL MTBM DATA LOADER")
    print("=" * 60)

    # ============================================
    # EDIT THIS SECTION FOR YOUR DATA
    # ============================================

    # Option 1: Excel file
    USE_EXCEL = True  # Set to True if you have Excel file
    excel_file = 'MTBM_Data_2024.xlsx'  # Your Excel file name
    sheet_name = 'Sheet1'  # Your sheet name

    # Option 2: CSV file
    USE_CSV = False  # Set to True if you have CSV file
    csv_file = 'mtbm_data.csv'  # Your CSV file name

    # ============================================

    if USE_EXCEL:
        df = load_mtbm_excel(excel_file, sheet_name)
    elif USE_CSV:
        df = load_mtbm_csv(csv_file)
    else:
        print("\n❌ Please set USE_EXCEL or USE_CSV to True")
        print("and specify your file name in the script.")
        sys.exit(1)

    if df is None:
        print("\n❌ Failed to load data. Please check:")
        print("   1. File path is correct")
        print("   2. Column names are mapped correctly")
        print("   3. File is not corrupted")
        sys.exit(1)

    # Save cleaned data
    output_file = 'cleaned_mtbm_data.csv'
    df.to_csv(output_file, index=False)

    print(f"\n💾 Saved cleaned data to: {output_file}")

    # Show data summary
    print("\n" + "=" * 60)
    print("📊 DATA SUMMARY")
    print("=" * 60)

    print("\nFirst 5 rows:")
    print(df.head())

    if 'advance_rate' in df.columns:
        print("\n📈 Advance Rate Statistics:")
        print(df['advance_rate'].describe())

    print("\n" + "=" * 60)
    print("✅ SUCCESS!")
    print("=" * 60)
    print("\nYour data is ready for machine learning!")
    print(f"Next step: Run 'python train_with_real_data.py'")


if __name__ == "__main__":
    main()
