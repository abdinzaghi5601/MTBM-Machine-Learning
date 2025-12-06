# Multi-Protocol MTBM Analysis Guide

Complete guide for using the MTBM framework with different AVN protocols (800, 1200, 2400, 3000).

## Table of Contents
1. [Overview](#overview)
2. [Supported Protocols](#supported-protocols)
3. [Quick Start](#quick-start)
4. [Protocol Differences](#protocol-differences)
5. [Usage Examples](#usage-examples)
6. [File Organization](#file-organization)
7. [Advanced Usage](#advanced-usage)

---

## Overview

The MTBM ML Framework now supports **all AVN protocols** with a unified interface. Each protocol has:

- **Different parameters** - Some protocols have more measurements than others
- **Different thresholds** - Quality standards vary by protocol capability
- **Different ranges** - Operating ranges adjusted for each protocol
- **Separate outputs** - Each protocol gets its own output directory

### Architecture

```
You provide:                Framework handles:              You get:
-----------                 ------------------              --------
Protocol Type  ────────>    Config Selection   ────────>    Protocol-specific
(AVN800, etc)               Parameter Mapping               outputs

Your Data      ────────>    Validation         ────────>    Validated results
(CSV file)                  Range Checking

                            Plotting           ────────>    Customized plots
                            Analysis                        Reports
```

---

## Supported Protocols

### AVN 800 - Basic Protocol
**Parameters**: 15+
- Core survey measurements
- Basic steering control
- Essential operational parameters

**Best For**:
- Standard tunneling projects
- Basic deviation tracking
- Cost-effective monitoring

**Key Features**:
- ✅ Position tracking
- ✅ Basic deviation measurement
- ✅ Cylinder control
- ✅ Advance speed, pressure, RPM

### AVN 1200 - Enhanced Protocol
**Parameters**: 18+
- All AVN 800 features PLUS:
- Yaw and pitch measurements
- Temperature monitoring
- Enhanced survey capabilities

**Best For**:
- Projects requiring precise orientation
- Temperature-sensitive environments
- Enhanced quality control

**Key Features**:
- ✅ All AVN 800 features
- ✅ Yaw/Pitch orientation
- ✅ ELS/MWD temperature
- ✅ Enhanced survey modes

### AVN 2400 - Advanced Protocol
**Parameters**: 22+
- All AVN 1200 features PLUS:
- Drill head specific measurements
- Enhanced force monitoring
- Reel angle tracking

**Best For**:
- Complex geology projects
- High-precision requirements
- Advanced steering analysis

**Key Features**:
- ✅ All AVN 1200 features
- ✅ Drill head deviations (separate from machine)
- ✅ Total force and interjack force
- ✅ Reel angle measurement
- ✅ Enhanced steering analysis

### AVN 3000 - Full Feature Protocol
**Parameters**: 23+
- All AVN 2400 features PLUS:
- Survey mode tracking
- Interjack activation monitoring
- Tighter quality thresholds

**Best For**:
- Mission-critical projects
- Research and development
- Maximum precision requirements
- Complete operational monitoring

**Key Features**:
- ✅ All AVN 2400 features
- ✅ Survey mode (ELS/ELS-HWL/GNS)
- ✅ Active interjack tracking
- ✅ Tighter deviation thresholds (20/40/60mm vs 25/50/75mm)
- ✅ Most comprehensive analysis

---

## Quick Start

### Method 1: Command Line Interface (Easiest)

```bash
cd MTBM-Machine-Learning

# Analyze your data with AVN 2400 protocol:
python analyze_protocol.py --protocol AVN2400 --data ../data/raw/my_data.csv

# Generate sample data for AVN 800:
python analyze_protocol.py --protocol AVN800 --generate-sample

# Analyze with AVN 3000 and 5000 samples:
python analyze_protocol.py --protocol AVN3000 --generate-sample --samples 5000
```

### Method 2: Python Script

```python
from analyze_protocol import ProtocolAnalyzer
import pandas as pd

# Create analyzer for your protocol
analyzer = ProtocolAnalyzer('AVN2400')

# Load your data
df = pd.read_csv('my_data.csv')

# Run complete analysis
analyzer.analyze(df)

# Outputs automatically saved to: outputs/AVN2400/
```

### Method 3: Interactive Python

```python
from protocol_configs import get_protocol_config
from analyze_protocol import ProtocolAnalyzer

# Explore protocol configuration
config = get_protocol_config('AVN1200')
print(f"Protocol: {config.protocol_name}")
print(f"Parameters: {config.get_parameter_names()}")
print(f"Thresholds: {config.deviation_thresholds}")

# Run analysis
analyzer = ProtocolAnalyzer('AVN1200')
df = analyzer.generate_sample_data(n_samples=1000)
analyzer.analyze(df)
```

---

## Protocol Differences

### Parameter Comparison

| Parameter | AVN 800 | AVN 1200 | AVN 2400 | AVN 3000 |
|-----------|---------|----------|----------|----------|
| Tunnel Length | ✅ | ✅ | ✅ | ✅ |
| Machine Deviations (H/V) | ✅ | ✅ | ✅ | ✅ |
| Drill Head Deviations | ❌ | ❌ | ✅ | ✅ |
| Yaw/Pitch | ❌ | ✅ | ✅ | ✅ |
| Reel Angle | ❌ | ❌ | ✅ | ✅ |
| Temperature | ❌ | ✅ | ✅ | ✅ |
| Cylinder Strokes (4x) | ✅ | ✅ | ✅ | ✅ |
| Total Force | ❌ | ❌ | ✅ | ✅ |
| Interjack Force | ❌ | ❌ | ✅ | ✅ |
| Advance Speed | ✅ | ✅ | ✅ | ✅ |
| Working Pressure | ✅ | ✅ | ✅ | ✅ |
| Revolution RPM | ✅ | ✅ | ✅ | ✅ |
| Earth Pressure | ✅ | ✅ | ✅ | ✅ |
| Survey Mode | ❌ | ❌ | ❌ | ✅ |
| Active Interjack | ❌ | ❌ | ❌ | ✅ |

### Quality Threshold Differences

| Threshold | AVN 800/1200/2400 | AVN 3000 |
|-----------|-------------------|----------|
| Excellent | ≤ 25mm | ≤ 20mm |
| Good | ≤ 50mm | ≤ 40mm |
| Poor | > 75mm | > 60mm |

**Reason**: AVN 3000 has tighter control, enabling higher precision.

### Operating Range Differences

#### Advance Speed (mm/min)

| Protocol | Normal Range | Critical Thresholds |
|----------|--------------|---------------------|
| AVN 800/1200/2400 | 15-45 | 10-55 |
| AVN 3000 | 20-40 | 10-55 |

#### Earth Pressure (bar)

| Protocol | Normal Range | Critical Thresholds |
|----------|--------------|---------------------|
| AVN 800/1200/2400 | 8-26 | 5-30 |
| AVN 3000 | 10-24 | 5-30 |

---

## Usage Examples

### Example 1: Analyze Real Data from AVN 800

```bash
# You have: field_data_avn800.csv in data/raw/

cd MTBM-Machine-Learning
python analyze_protocol.py \
    --protocol AVN800 \
    --data ../data/raw/field_data_avn800.csv

# Results saved to: outputs/AVN800/
```

**Output Structure**:
```
outputs/AVN800/
├── plots/
│   ├── mtbm_time_series_overview.png
│   ├── mtbm_deviation_analysis.png
│   ├── mtbm_performance_dashboard.png
│   └── mtbm_correlation_matrix.png
├── data/
│   └── AVN_800_data_20241123_150000.csv
└── reports/
    └── analysis_report_20241123_150000.txt
```

### Example 2: Compare Two Protocols

```bash
# Analyze same data with different protocols:

# AVN 1200 analysis
python analyze_protocol.py --protocol AVN1200 --data my_data.csv

# AVN 2400 analysis
python analyze_protocol.py --protocol AVN2400 --data my_data.csv

# Now compare:
# outputs/AVN1200/plots/ vs outputs/AVN2400/plots/
```

**Use Case**: See if AVN 2400's additional measurements provide more insight.

### Example 3: Generate Test Data for Each Protocol

```bash
# Generate sample data to see what each protocol looks like:

python analyze_protocol.py --protocol AVN800 --generate-sample --samples 1000
python analyze_protocol.py --protocol AVN1200 --generate-sample --samples 1000
python analyze_protocol.py --protocol AVN2400 --generate-sample --samples 1000
python analyze_protocol.py --protocol AVN3000 --generate-sample --samples 1000

# Each creates its own output directory with sample visualizations
```

### Example 4: Validate Data Before Analysis

```bash
# Check if your data is compatible with AVN 2400:

python analyze_protocol.py \
    --protocol AVN2400 \
    --data my_data.csv \
    --validate-only

# Output shows:
# - Missing parameters
# - Values outside normal ranges
# - Values outside absolute limits
```

### Example 5: Data Analysis Only (No Plots)

```bash
# Useful for automated processing or large datasets:

python analyze_protocol.py \
    --protocol AVN3000 \
    --data my_data.csv \
    --no-plots

# Generates CSV and text report only (much faster)
```

### Example 6: Python Script for Batch Processing

```python
from analyze_protocol import ProtocolAnalyzer
import pandas as pd
from pathlib import Path

# Process multiple files with AVN 2400
data_files = Path('../data/raw').glob('*.csv')

for data_file in data_files:
    print(f"\n Processing: {data_file.name}")

    analyzer = ProtocolAnalyzer('AVN2400')
    df = pd.read_csv(data_file)

    # Validate first
    validation = analyzer.validate_data(df)

    if validation['valid']:
        analyzer.analyze(df, save_plots=True)
    else:
        print(f"❌ Skipping {data_file.name} - validation failed")
```

### Example 7: Custom Analysis with Protocol Config

```python
from protocol_configs import get_protocol_config
from analyze_protocol import ProtocolAnalyzer
import pandas as pd

# Get AVN 3000 configuration
config = get_protocol_config('AVN3000')

# Create analyzer
analyzer = ProtocolAnalyzer('AVN3000')

# Generate data
df = analyzer.generate_sample_data(n_samples=2000)

# Check specific parameter against protocol thresholds
advance_speed = df['advance_speed_mm_min']
speed_config = config.parameters['advance_speed_mm_min']

print(f"Speed range: {advance_speed.min():.1f} - {advance_speed.max():.1f}")
print(f"Normal range: {speed_config.normal_min} - {speed_config.normal_max}")

# Calculate percentage within normal range
in_range = (advance_speed >= speed_config.normal_min) & \
           (advance_speed <= speed_config.normal_max)
pct_in_range = (in_range.sum() / len(advance_speed)) * 100

print(f"Within normal range: {pct_in_range:.1f}%")

# Run analysis
analyzer.analyze(df)
```

---

## File Organization

### Protocol-Specific Output Structure

```
outputs/
├── AVN800/
│   ├── plots/
│   ├── data/
│   └── reports/
├── AVN1200/
│   ├── plots/
│   ├── data/
│   └── reports/
├── AVN2400/
│   ├── plots/
│   ├── data/
│   └── reports/
└── AVN3000/
    ├── plots/
    ├── data/
    └── reports/
```

### Benefits of Separation

1. **No conflicts** - Run multiple protocols without overwriting files
2. **Easy comparison** - Side-by-side protocol analysis
3. **Clear organization** - Know which protocol generated which output
4. **Version control** - Track protocol-specific results over time

---

## Advanced Usage

### Creating a Custom Protocol

```python
# In protocol_configs.py

class AVNCustomConfig(ProtocolConfig):
    """Your custom protocol configuration"""

    def __init__(self):
        super().__init__("AVN Custom")

        # Start from existing protocol
        avn2400 = AVN2400Config()
        for param in avn2400.parameters.values():
            self.add_parameter(param)

        # Add your custom parameters
        self.add_parameter(ParameterConfig(
            name='custom_sensor_reading',
            display_name='Custom Sensor',
            unit='units',
            min_value=0,
            max_value=100,
            normal_min=20,
            normal_max=80
        ))

        # Customize thresholds
        self.deviation_thresholds = {
            'excellent': 15,  # Very tight
            'good': 30,
            'poor': 50
        }

# Update factory function
def get_protocol_config(protocol_name: str):
    protocol_map = {
        # ... existing protocols ...
        'CUSTOM': AVNCustomConfig,
    }
    # ... rest of function
```

### Modifying Protocol Thresholds

```python
from analyze_protocol import ProtocolAnalyzer

# Create analyzer
analyzer = ProtocolAnalyzer('AVN2400')

# Modify thresholds
analyzer.config.deviation_thresholds = {
    'excellent': 20,  # Changed from 25
    'good': 45,       # Changed from 50
    'poor': 70        # Changed from 75
}

# Modify parameter ranges
speed_param = analyzer.config.parameters['advance_speed_mm_min']
speed_param.normal_min = 18  # Changed from 15
speed_param.normal_max = 42  # Changed from 45

# Run analysis with custom thresholds
df = analyzer.generate_sample_data()
analyzer.analyze(df)
```

### Protocol-Aware Data Loading

```python
from analyze_protocol import ProtocolAnalyzer
import pandas as pd

def load_protocol_data(file_path, protocol_name):
    """
    Load data and ensure it matches protocol expectations
    """
    # Create analyzer for protocol
    analyzer = ProtocolAnalyzer(protocol_name)

    # Load raw data
    df = pd.read_csv(file_path)

    # Get required parameters for this protocol
    required_params = analyzer.config.get_parameter_names()

    # Check which parameters are missing
    missing = [p for p in required_params if p not in df.columns]

    if missing:
        print(f"Warning: Missing {len(missing)} parameters for {protocol_name}")
        print(f"Missing: {missing[:5]}...")  # Show first 5

    # Keep only columns that exist in protocol
    available_cols = ['timestamp', 'date', 'time'] + [
        col for col in df.columns if col in required_params
    ]

    df = df[available_cols]

    return df, analyzer

# Usage:
df, analyzer = load_protocol_data('my_data.csv', 'AVN2400')
analyzer.analyze(df)
```

### Batch Protocol Comparison

```python
from analyze_protocol import ProtocolAnalyzer
from protocol_configs import SUPPORTED_PROTOCOLS
import pandas as pd

# Load your data once
df_original = pd.read_csv('my_data.csv')

# Analyze with each protocol
results = {}

for protocol in SUPPORTED_PROTOCOLS:
    print(f"\n{'='*60}")
    print(f"Analyzing with {protocol}")
    print(f"{'='*60}")

    analyzer = ProtocolAnalyzer(protocol)

    # Validate
    validation = analyzer.validate_data(df_original.copy())
    results[protocol] = validation

    # If valid, analyze
    if validation['valid']:
        analyzer.analyze(df_original.copy(), save_plots=True)

# Summary
print("\n\nPROTOCOL COMPATIBILITY SUMMARY")
print("="*60)
for protocol, validation in results.items():
    status = "✅ VALID" if validation['valid'] else "❌ INVALID"
    print(f"{protocol:12} {status:12} "
          f"Errors: {len(validation['errors']):2} "
          f"Warnings: {len(validation['warnings']):2}")
```

---

## Common Scenarios

### Scenario 1: Upgrading from AVN 800 to AVN 2400

**Situation**: You've been using AVN 800, now upgrading equipment to AVN 2400.

**Solution**:
```bash
# Analyze old data with AVN 800
python analyze_protocol.py --protocol AVN800 --data old_data.csv

# Analyze new data with AVN 2400
python analyze_protocol.py --protocol AVN2400 --data new_data.csv

# Compare outputs:
# outputs/AVN800/ vs outputs/AVN2400/
```

**What to Expect**:
- AVN 2400 provides drill head deviations (new!)
- Force measurements more detailed
- Additional parameters for better analysis

### Scenario 2: Data from Unknown Protocol

**Situation**: You have data but aren't sure which protocol it came from.

**Solution**:
```bash
# Try validating with each protocol
python analyze_protocol.py --protocol AVN800 --data mystery_data.csv --validate-only
python analyze_protocol.py --protocol AVN1200 --data mystery_data.csv --validate-only
python analyze_protocol.py --protocol AVN2400 --data mystery_data.csv --validate-only
python analyze_protocol.py --protocol AVN3000 --data mystery_data.csv --validate-only

# The one with fewest errors/most parameters found is likely your protocol
```

### Scenario 3: Mixed Protocol Project

**Situation**: Project started with AVN 1200, later switched to AVN 3000.

**Solution**:
```python
# Separate your data by date/protocol
df_early = pd.read_csv('project_phase1.csv')  # AVN 1200 data
df_later = pd.read_csv('project_phase2.csv')  # AVN 3000 data

# Analyze each with appropriate protocol
analyzer_1200 = ProtocolAnalyzer('AVN1200')
analyzer_1200.analyze(df_early)

analyzer_3000 = ProtocolAnalyzer('AVN3000')
analyzer_3000.analyze(df_later)

# Compare results in:
# outputs/AVN1200/ vs outputs/AVN3000/
```

---

## Troubleshooting

### "Unknown protocol" Error

**Error**: `ValueError: Unknown protocol: AVN 2400`

**Solution**: Use exact protocol names without spaces:
```bash
# ❌ Wrong
python analyze_protocol.py --protocol "AVN 2400"

# ✅ Correct
python analyze_protocol.py --protocol AVN2400
```

### Missing Parameters

**Warning**: `Parameter 'yaw_mm_per_m' not found in data`

**Meaning**: Your data doesn't include this parameter (common if you have data from a lower protocol).

**Solution**:
- If intentional: Ignore warning, analysis continues with available parameters
- If error: Check your data file columns, ensure proper naming

### Values Outside Range

**Error**: `earth_pressure_01_bar: Values outside allowed range`

**Common Cause**: Decimal point issue (earth pressure 160 instead of 16.0)

**Solution**: Check data preprocessing, apply scaling if needed:
```python
df['earth_pressure_01_bar'] = df['earth_pressure_01_bar'] / 10
```

### Protocol Mismatch

**Warning**: Many parameters outside normal range

**Likely Cause**: Using wrong protocol for your data

**Solution**: Try different protocols or validate with each to find best match

---

## Best Practices

### 1. Choose the Right Protocol

- **Use the protocol your equipment actually has**
- Don't use AVN 3000 config for AVN 800 data
- Validate your data against protocol first

### 2. Organize Your Data

```
data/
├── raw/
│   ├── avn800/
│   ├── avn1200/
│   ├── avn2400/
│   └── avn3000/
```

### 3. Name Files Descriptively

```
project_phase1_avn1200_20241120.csv  ✅ Good
data.csv                              ❌ Bad
```

### 4. Validate Before Analyzing

```bash
# Always validate first
python analyze_protocol.py --protocol AVN2400 --data my_data.csv --validate-only

# Then analyze
python analyze_protocol.py --protocol AVN2400 --data my_data.csv
```

### 5. Document Protocol Changes

Keep a log when switching protocols:
```
Project Log:
- 2024-01-01 to 2024-03-15: AVN 1200
- 2024-03-16 onwards: AVN 3000 (equipment upgrade)
```

---

## Summary

### Quick Command Reference

```bash
# Generate sample data
python analyze_protocol.py --protocol AVN800 --generate-sample

# Analyze real data
python analyze_protocol.py --protocol AVN2400 --data my_data.csv

# Validate only
python analyze_protocol.py --protocol AVN3000 --data my_data.csv --validate-only

# Skip plots (faster)
python analyze_protocol.py --protocol AVN1200 --data my_data.csv --no-plots

# Custom sample count
python analyze_protocol.py --protocol AVN2400 --generate-sample --samples 5000
```

### Protocol Selection Guide

| Your Situation | Recommended Protocol |
|---------------|---------------------|
| Basic monitoring | AVN 800 |
| Need orientation data | AVN 1200 |
| Complex geology | AVN 2400 |
| Maximum precision | AVN 3000 |
| Research project | AVN 3000 |

---

**Document Version**: 1.0
**Last Updated**: November 2024
**Supported Protocols**: AVN 800, 1200, 2400, 3000
