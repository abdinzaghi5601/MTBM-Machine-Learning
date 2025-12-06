# Steering Calculator - Usage Guide

## Overview

The steering calculator system has been consolidated into a clean, professional structure with two main components:

1. **`steering_calculator.py`** - Core calculation library (842 lines)
2. **`steering_cli.py`** - Interactive command-line interface (246 lines)

**Supported Cylinder Systems:** 3-Cylinder, 4-Cylinder, and 6-Cylinder

All formulas extracted from `Steer-cyl-cal-rev8..xls` and verified through comprehensive testing.

## Quick Start

### Method 1: Interactive CLI (Easiest for Field Use)

```bash
python3 steering_cli.py
```

This launches an interactive mode that guides you through:
- Machine parameter input
- Cylinder readings
- Target steering state
- Comprehensive report generation
- Optional file export

### Method 2: Python Code (Most Flexible)

```python
from steering_calculator import SteeringCalculator, MachineParameters, CylinderReadings

# Setup
params = MachineParameters(
    num_cylinders=3,
    stroke=50.0,
    mounting_diameter=715.0,
    pipe_length=3000.0
)
calc = SteeringCalculator(params)

# Analyze current state
readings = CylinderReadings(cylinder_1=28.69, cylinder_2=31.52, cylinder_3=21.79)
analysis = calc.analyze_current_state(readings)
print(f"Current Pitch: {analysis['current_steering']['pitch']} mm/m")

# Plan correction
correction_plan = calc.plan_correction(
    current_pitch=10.3,
    current_yaw=-9.7,
    target_pitch=-4.5,
    target_yaw=16.5
)

# Generate report
report = calc.generate_report(analysis, correction_plan)
print(report)
```

### Method 3: Quick Functions (For Simple Calculations)

```python
from steering_calculator import quick_calculate, quick_reverse

# Calculate cylinder positions from pitch/yaw
cylinders = quick_calculate(pitch=-5, yaw=10)
print(cylinders)
# Output: {'cylinder_1': 23.21, 'cylinder_2': 28.73, 'cylinder_3': 21.06}

# Calculate pitch/yaw from cylinder positions
pitch, yaw = quick_reverse([20.0, 32.0, 30.0])
print(f"Pitch: {pitch}, Yaw: {yaw}")
# Output: Pitch: -13.99, Yaw: 3.23
```

## Core Features

### SteeringCalculator Class

The main class provides comprehensive functionality:

#### 1. Forward Calculations (Pitch/Yaw → Cylinders)

```python
from steering_calculator import SteeringCommand

steering = SteeringCommand(pitch=-4.5, yaw=16.5)
cylinders = calc.calculate_cylinders(steering)
```

Supports 3, 4, and 6-cylinder systems automatically based on configuration.

#### 2. Reverse Calculations (Cylinders → Pitch/Yaw)

```python
readings = CylinderReadings(cylinder_1=20.0, cylinder_2=32.0, cylinder_3=30.0)
steering = calc.calculate_steering(readings)
print(f"Pitch: {steering.pitch}, Yaw: {steering.yaw}")
```

#### 3. Current State Analysis

```python
analysis = calc.analyze_current_state(readings)
# Returns: system_type, parameters, cylinder_readings, current_steering,
#          gradient_analysis, cylinder_status
```

#### 4. Correction Planning

```python
plan = calc.plan_correction(
    current_pitch=10.3,
    current_yaw=-9.7,
    target_pitch=-4.5,
    target_yaw=16.5
)
# Returns: current_state, target_state, required_correction,
#          cylinder_positions, correction_per_pipe, expected_result,
#          feasibility, warnings
```

#### 5. Report Generation

```python
report = calc.generate_report(analysis, correction_plan)
print(report)

# Save to file
with open('report.txt', 'w') as f:
    f.write(report)
```

## Data Structures

### MachineParameters

Configuration for your MTBM:

```python
params = MachineParameters(
    num_cylinders=3,          # 3, 4, or 6
    stroke=50.0,              # mm
    mounting_diameter=715.0,  # mm
    pipe_length=3000.0,       # mm
    vertical_angle=1.49,      # mm/m
    laser_gradient=0.00149,
    dist_head_to_target=2331.0,  # mm
    length_steering_head=991.0,  # mm
    target_above_axis=140.0      # mm
)
```

### SteeringCommand

Pitch and yaw corrections:

```python
steering = SteeringCommand(
    pitch=-4.5,  # mm/m (positive = up, negative = down)
    yaw=16.5     # mm/m (positive = right, negative = left)
)
```

### CylinderReadings

Current cylinder positions:

```python
# 3-cylinder system
readings = CylinderReadings(
    cylinder_1=28.69,
    cylinder_2=31.52,
    cylinder_3=21.79
)

# 4-cylinder system
readings = CylinderReadings(
    cylinder_1=25.0,
    cylinder_2=30.0,
    cylinder_3=25.0,
    cylinder_4=20.0
)

# 6-cylinder system
readings = CylinderReadings(
    cylinder_1=25.0, cylinder_2=27.0, cylinder_3=23.0,
    cylinder_4=25.0, cylinder_5=28.0, cylinder_6=22.0
)
```

## Advanced Usage

### Batch Processing

Process multiple calculations in a loop:

```python
from steering_calculator import SteeringCalculator, MachineParameters, CylinderReadings

params = MachineParameters()
calc = SteeringCalculator(params)

# List of cylinder readings to process
readings_list = [
    [28.69, 31.52, 21.79],
    [20.0, 32.0, 30.0],
    [25.0, 25.0, 25.0]
]

for readings_data in readings_list:
    readings = CylinderReadings(
        cylinder_1=readings_data[0],
        cylinder_2=readings_data[1],
        cylinder_3=readings_data[2]
    )
    steering = calc.calculate_steering(readings)
    print(f"Pitch: {steering.pitch:7.2f}, Yaw: {steering.yaw:7.2f}")
```

### Custom Validation

Add your own validation logic:

```python
analysis = calc.analyze_current_state(readings)

# Check for extreme pitch
if abs(analysis['current_steering']['pitch']) > 30:
    print("WARNING: Extreme pitch detected!")

# Check cylinder range
for cyl, value in analysis['cylinder_readings'].items():
    if value < 10 or value > 40:
        print(f"WARNING: {cyl} at {value}mm is near limit")
```

### Integration with Data Logging

```python
import json
from datetime import datetime

# Perform analysis
analysis = calc.analyze_current_state(readings)
correction_plan = calc.plan_correction(10.3, -9.7, -4.5, 16.5)

# Create log entry
log_entry = {
    'timestamp': datetime.now().isoformat(),
    'analysis': analysis,
    'correction_plan': correction_plan
}

# Save to JSON
with open('steering_log.json', 'a') as f:
    f.write(json.dumps(log_entry) + '\n')
```

## CLI Usage

### Interactive Mode

```bash
$ python3 steering_cli.py
```

Follow the prompts:
1. Enter machine parameters (or use defaults)
2. Enter current cylinder readings
3. Enter target pitch/yaw
4. View comprehensive report
5. Optionally save to file

### Quick Mode

For fast calculations without full analysis:

```bash
$ python3 steering_cli.py
# Select option 2 for Quick Mode
```

Quick mode provides:
- Fast cylinder calculations from pitch/yaw
- Fast pitch/yaw calculations from cylinders
- Minimal prompts
- Standard machine parameters

## System Support

### 3-Cylinder System (120° spacing)

Most common configuration:
- Cylinder 1: 0° (Top - 12 o'clock)
- Cylinder 2: 120° (Lower right - 4 o'clock)
- Cylinder 3: 240° (Lower left - 8 o'clock)

### 4-Cylinder System (90° spacing)

- Cylinder 1: 0° (Top)
- Cylinder 2: 90° (Right)
- Cylinder 3: 180° (Bottom)
- Cylinder 4: 270° (Left)

### 6-Cylinder System (60° spacing)

- Cylinders at 60° intervals starting from top (0°)

## Validation and Safety

The calculator includes comprehensive validation:

### Stroke Limit Checking

Automatically checks if cylinder positions are within stroke limits (0 to max stroke).

### Feasibility Analysis

Before applying corrections, checks if all cylinders will remain within limits.

### Warning System

Generates warnings for:
- Cylinders near minimum (<5mm)
- Cylinders near maximum (>stroke-5mm)
- Cylinders out of range
- Extreme pitch corrections (>50 mm/m)
- Extreme yaw corrections (>50 mm/m)

### Example Warning Output

```
WARNINGS:
  • cylinder_2 very near maximum stroke (47.50mm)
  • Very high yaw correction (55.0 mm/m)
```

## Typical Workflow

### 1. Field Operations

```python
# Initialize with your machine specs
params = MachineParameters(
    num_cylinders=3,
    stroke=50.0,
    mounting_diameter=715.0
)
calc = SteeringCalculator(params)

# Read current cylinder positions from SCADA
readings = CylinderReadings(cylinder_1=28.69, cylinder_2=31.52, cylinder_3=21.79)

# Analyze current state
analysis = calc.analyze_current_state(readings)
print(f"Current: Pitch={analysis['current_steering']['pitch']}, "
      f"Yaw={analysis['current_steering']['yaw']}")

# Plan correction to target
plan = calc.plan_correction(
    current_pitch=analysis['current_steering']['pitch'],
    current_yaw=analysis['current_steering']['yaw'],
    target_pitch=0.0,
    target_yaw=0.0
)

# Check feasibility
if plan['feasibility']['is_feasible']:
    print("Correction is feasible")
    print(f"Set cylinders to: {plan['cylinder_positions']}")
else:
    print(f"WARNING: {plan['feasibility']['reason']}")
```

### 2. Offline Analysis

```python
# Generate full report for review
report = calc.generate_report(analysis, plan)

# Save for documentation
with open(f"steering_report_{datetime.now():%Y%m%d_%H%M}.txt", 'w') as f:
    f.write(report)
```

## Performance

- Calculations are nearly instantaneous (<1ms)
- No external dependencies required (pure Python + standard library)
- Suitable for real-time SCADA integration
- Memory efficient

## Troubleshooting

### Import Errors

```python
# Make sure steering_calculator.py is in the same directory
import sys
sys.path.append('/path/to/calculator')
from steering_calculator import *
```

### Unexpected Results

1. Check sign conventions:
   - Positive pitch = UP
   - Negative pitch = DOWN
   - Positive yaw = RIGHT
   - Negative yaw = LEFT

2. Verify cylinder numbering matches your machine

3. Confirm mounting diameter and stroke are correct

### Validation Failures

If corrections fail feasibility checks:
- Break large corrections into smaller steps
- Check if cylinders are already at limits
- Verify stroke and mounting diameter parameters

## File Structure

```
steering_calculator.py    - Core calculation library
steering_cli.py          - Interactive command-line interface
USAGE_GUIDE.md          - This file
PROJECT_SUMMARY.md      - Project overview and consolidation notes
QUICK_REFERENCE.md      - Field reference card
```

## Migration from Old Files

If you were using the old `interactive_calculator.py`:

**Old:**
```python
from interactive_calculator import SteeringAnalyzer
analyzer = SteeringAnalyzer(params)
```

**New:**
```python
from steering_calculator import SteeringCalculator
calc = SteeringCalculator(params)
```

All method names remain the same. The `SteeringCalculator` class now includes all features from both old files.

## Support

For issues or questions:
1. Check this guide
2. Review QUICK_REFERENCE.md for formulas
3. See PROJECT_SUMMARY.md for technical details
4. Run example code in `steering_calculator.py` (python3 steering_calculator.py)

## Version History

**v2.0 (Consolidated)** - December 2024
- Merged interactive_calculator.py and steering_calculator.py
- Created separate CLI interface
- Added quick calculation functions
- Enhanced documentation
- Improved code organization

**v1.0 (Initial)** - December 2024
- Initial reverse-engineering from Excel
- Basic calculations and analysis
- Report generation
