# MTBM Code Structure & Understanding Guide

A comprehensive guide to understanding the codebase, how it works, and how to use it effectively.

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Main Components](#main-components)
3. [Data Flow](#data-flow)
4. [Key Scripts Explained](#key-scripts-explained)
5. [How to Run](#how-to-run)
6. [Customization Guide](#customization-guide)

---

## Project Architecture

```
ML for Tunneling/
│
├── MTBM-Machine-Learning/          # Main ML framework
│   ├── mtbm_comprehensive_plotting.py   # Main visualization script
│   ├── generate_mtbm_graphs.py          # Graph generation utilities
│   └── plot_real_mtbm_data.py           # Real data plotting
│
├── data/                            # Data directory
│   ├── raw/                         # Original data files
│   └── processed/                   # Cleaned, processed data
│       └── mtbm_comprehensive_data.csv  # Generated here
│
├── outputs/                         # All outputs go here
│   ├── plots/                       # PNG visualizations
│   │   ├── mtbm_time_series_overview.png
│   │   ├── mtbm_deviation_analysis.png
│   │   ├── mtbm_performance_dashboard.png
│   │   └── mtbm_correlation_matrix.png
│   ├── reports/                     # Text reports
│   ├── models/                      # Trained ML models
│   └── logs/                        # Execution logs
│
└── docs/                            # Documentation
    ├── PLOT_INTERPRETATION_GUIDE.md
    └── CODE_STRUCTURE_GUIDE.md (this file)
```

---

## Main Components

### 1. MTBMComprehensivePlotter Class

**Location**: `MTBM-Machine-Learning/mtbm_comprehensive_plotting.py`

**Purpose**: Central class that handles all data generation, processing, and visualization.

#### Key Responsibilities:
- Generate synthetic MTBM operational data
- Create all 4 types of visualizations
- Manage output directories
- Calculate derived metrics (efficiency, total deviation)

#### Class Structure:

```python
class MTBMComprehensivePlotter:
    def __init__(self, base_dir=None):
        # Sets up directory structure
        # Defines parameter groups
        # Configures color schemes

    def generate_synthetic_mtbm_data(self, n_samples=1000):
        # Creates realistic MTBM operational data
        # Returns: pandas DataFrame

    def plot_time_series_overview(self, df, save_plots=True):
        # Creates 6x4 grid of time series plots
        # Saves to: outputs/plots/mtbm_time_series_overview.png

    def plot_deviation_analysis(self, df, save_plots=True):
        # Creates 2x2 deviation analysis plots
        # Saves to: outputs/plots/mtbm_deviation_analysis.png

    def plot_performance_dashboard(self, df, save_plots=True):
        # Creates 2x3 performance dashboard
        # Saves to: outputs/plots/mtbm_performance_dashboard.png

    def plot_correlation_matrix(self, df, save_plots=True):
        # Creates correlation heatmap
        # Saves to: outputs/plots/mtbm_correlation_matrix.png

    def generate_comprehensive_report(self, df):
        # Prints text-based analysis report to console
```

### 2. Parameter Groups

The framework organizes 23+ parameters into logical groups:

```python
self.parameters = {
    'temporal': {
        # Date, time information
    },
    'survey_position': {
        # Location and deviation data
        # tunnel_length_m, hor_deviation_machine_mm, etc.
    },
    'survey_orientation': {
        # Angle and orientation
        # yaw_mm_per_m, pitch_mm_per_m, reel_degree, etc.
    },
    'steering_control': {
        # Hydraulic cylinder positions
        # cylinder_01_stroke_mm through cylinder_04_stroke_mm
        # total_force_kn
    },
    'operational': {
        # Operation parameters
        # advance_speed_mm_min, interjack_force_kn, etc.
    },
    'cutter_wheel': {
        # Cutting parameters
        # working_pressure_bar, revolution_rpm, earth_pressure_01_bar
    }
}
```

### 3. Color Schemes

Each parameter group has a dedicated color for visual consistency:

```python
self.color_schemes = {
    'temporal': '#1f77b4',          # Blue
    'survey_position': '#ff7f0e',   # Orange
    'survey_orientation': '#2ca02c', # Green
    'steering_control': '#d62728',   # Red
    'operational': '#9467bd',        # Purple
    'cutter_wheel': '#8c564b'        # Brown
}
```

---

## Data Flow

### Overview of Data Processing Pipeline

```
1. Data Generation/Loading
   ↓
2. Data Cleaning & Processing
   ↓
3. Feature Engineering (derived metrics)
   ↓
4. Visualization Generation
   ↓
5. Report Generation
   ↓
6. File Output (CSV + PNG files)
```

### Detailed Flow

#### Step 1: Data Generation

```python
# In main() function
plotter = MTBMComprehensivePlotter()
df = plotter.generate_synthetic_mtbm_data(n_samples=1000)
```

**What happens**:
1. Creates timestamps at 30-minute intervals
2. Generates realistic parameter values:
   - Tunnel length: Cumulative progress
   - Deviations: Random walk with corrections
   - Cylinder strokes: Coordinated movement
   - Operational params: Realistic ranges
3. Returns pandas DataFrame with 1000 rows × 23+ columns

#### Step 2: Feature Engineering

```python
# Inside generate_synthetic_mtbm_data()
df['total_deviation_mm'] = np.sqrt(
    df['hor_deviation_machine_mm']**2 +
    df['vert_deviation_machine_mm']**2
)
df['drilling_efficiency'] = (
    df['advance_speed_mm_min'] / df['working_pressure_bar']
)
df['power_efficiency'] = (
    df['advance_speed_mm_min'] / df['revolution_rpm']
)
```

**Derived Metrics**:
- **Total Deviation**: Vector magnitude of horizontal + vertical
- **Drilling Efficiency**: How well pressure converts to progress
- **Power Efficiency**: How well rotation converts to progress

#### Step 3: Visualization

Each plot function:
1. Takes DataFrame as input
2. Creates matplotlib figure with subplots
3. Plots relevant parameters
4. Adds labels, legends, grid
5. Saves to `outputs/plots/` directory
6. Displays on screen

#### Step 4: Data Saving

```python
csv_path = plotter.processed_data_dir / 'mtbm_comprehensive_data.csv'
df.to_csv(csv_path, index=False)
```

Saves processed data for future use.

---

## Key Scripts Explained

### mtbm_comprehensive_plotting.py

**Purpose**: Complete MTBM analysis and visualization framework

**Key Functions**:

#### 1. `__init__(self, base_dir=None)`

```python
def __init__(self, base_dir=None):
    # Set up directory paths
    if base_dir is None:
        base_dir = Path(__file__).parent.parent

    self.plots_dir = base_dir / 'outputs' / 'plots'
    self.processed_data_dir = base_dir / 'data' / 'processed'

    # Create directories if they don't exist
    self.plots_dir.mkdir(parents=True, exist_ok=True)
    self.processed_data_dir.mkdir(parents=True, exist_ok=True)
```

**What it does**:
- Automatically finds project root directory
- Creates output directories if missing
- Safe to run multiple times

#### 2. `generate_synthetic_mtbm_data(self, n_samples=1000)`

```python
# Example of realistic data generation
tunnel_length = np.cumsum(np.random.uniform(0.5, 2.0, n_samples))
# Creates: [0.8, 2.1, 3.5, 5.2, ...] (cumulative meters)

hor_deviation_machine = np.cumsum(np.random.normal(0, 2, n_samples))
# Creates: [0.5, 1.2, -0.3, 2.1, ...] (random walk in mm)
```

**What it does**:
- Uses `cumsum()` for cumulative values (tunnel length, deviations)
- Uses `random.normal()` for parameters with expected means
- Uses `random.uniform()` for bounded ranges
- Creates realistic correlations between parameters

#### 3. `plot_time_series_overview(self, df, save_plots=True)`

```python
fig, axes = plt.subplots(6, 4, figsize=(20, 24))
# Creates 6 rows × 4 columns = 24 subplots

axes_flat = axes.flatten()
# Converts 2D array to 1D for easier indexing

plot_idx = 0
for param in position_params:
    axes_flat[plot_idx].plot(df['timestamp'], df[param], ...)
    plot_idx += 1
```

**What it does**:
- Creates large grid of subplots
- Plots each parameter group in sequence
- Uses consistent color scheme
- Hides unused subplot spaces

#### 4. `plot_deviation_analysis(self, df, save_plots=True)`

```python
# Scatter plot with tolerance circles
axes[0,0].scatter(df['hor_deviation_machine_mm'],
                  df['vert_deviation_machine_mm'],
                  c=df['tunnel_length_m'],  # Color by progress
                  cmap='viridis')

# Add tolerance circles
circle1 = plt.Circle((0, 0), 25, fill=False, color='green')
axes[0,0].add_patch(circle1)
```

**What it does**:
- Creates scatter plot colored by tunnel progress
- Adds reference circles for quality standards
- Shows relationship between deviation types

#### 5. `plot_correlation_matrix(self, df, save_plots=True)`

```python
# Calculate correlations
correlation_matrix = df[numerical_cols].corr()

# Create heatmap with mask (lower triangle only)
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, ...)
```

**What it does**:
- Computes correlation between all numeric parameters
- Shows only lower triangle (matrix is symmetric)
- Annotates each cell with correlation value
- Uses color scale (red=positive, blue=negative)

---

## How to Run

### Basic Usage

```bash
# Navigate to project directory
cd "/mnt/c/Users/abdul/Desktop/ML for Tunneling"

# Run the main script
cd MTBM-Machine-Learning
python mtbm_comprehensive_plotting.py
```

### Expected Output

```
MTBM Comprehensive Plotting Framework
====================================
Generating comprehensive MTBM operational data...
Saved comprehensive dataset: /path/to/data/processed/mtbm_comprehensive_data.csv

Generating comprehensive visualizations...
Creating time series overview plots...
Saved: /path/to/outputs/plots/mtbm_time_series_overview.png

Creating deviation analysis plots...
Saved: /path/to/outputs/plots/mtbm_deviation_analysis.png

Creating performance dashboard...
Saved: /path/to/outputs/plots/mtbm_performance_dashboard.png

Creating correlation analysis...
Saved: /path/to/outputs/plots/mtbm_correlation_matrix.png

Top 10 Strongest Correlations:
--------------------------------------------------
[... correlation analysis ...]

================================================================================
MTBM COMPREHENSIVE OPERATIONAL ANALYSIS REPORT
================================================================================
[... detailed report ...]
```

### Using with Real Data

Instead of synthetic data, load your actual MTBM data:

```python
# In main() function, replace:
df = plotter.generate_synthetic_mtbm_data(n_samples=1000)

# With:
df = pd.read_csv('../data/raw/your_actual_data.csv')

# Ensure column names match expected format:
# - timestamp (datetime)
# - tunnel_length_m (float)
# - hor_deviation_machine_mm (float)
# - etc.
```

---

## Customization Guide

### Adding New Parameters

**Step 1**: Add to parameter definition

```python
# In __init__()
self.parameters = {
    # ... existing groups ...
    'new_group': {
        'new_param': 'New Parameter Name'
    }
}
```

**Step 2**: Add to data generation

```python
# In generate_synthetic_mtbm_data()
new_param = np.random.uniform(min_val, max_val, n_samples)
data['new_param'] = new_param
```

**Step 3**: Add to visualization

```python
# In plot_time_series_overview()
new_params = ['new_param']
for param in new_params:
    axes_flat[plot_idx].plot(df['timestamp'], df[param], ...)
    plot_idx += 1
```

### Changing Quality Thresholds

```python
# In plot_deviation_analysis()
# Current thresholds: 25mm, 50mm, 75mm
# To change to: 20mm, 40mm, 60mm

circle1 = plt.Circle((0, 0), 20, ...)  # Was 25
circle2 = plt.Circle((0, 0), 40, ...)  # Was 50
circle3 = plt.Circle((0, 0), 60, ...)  # Was 75

axes[0,1].axhline(y=20, ...)  # Update all threshold lines
axes[0,1].axhline(y=40, ...)
axes[0,1].axhline(y=60, ...)
```

### Customizing Plot Appearance

```python
# Change figure size
fig, axes = plt.subplots(6, 4, figsize=(24, 28))  # Larger

# Change DPI (resolution)
plt.savefig(save_path, dpi=150, ...)  # Lower for smaller files
plt.savefig(save_path, dpi=600, ...)  # Higher for publications

# Change color scheme
self.color_schemes['survey_position'] = '#ff0000'  # Red instead of orange

# Change plot style
plt.style.use('seaborn-v0_8')  # Or 'ggplot', 'bmh', etc.
```

### Adding New Visualizations

```python
def plot_custom_analysis(self, df, save_plots=True):
    """
    Your custom analysis plot
    """
    print("Creating custom analysis...")

    fig, ax = plt.subplots(figsize=(12, 8))

    # Your plotting code here
    ax.plot(df['timestamp'], df['your_parameter'])
    ax.set_title('Custom Analysis')
    ax.set_xlabel('Time')
    ax.set_ylabel('Your Parameter')
    ax.grid(True)

    plt.tight_layout()

    if save_plots:
        save_path = self.plots_dir / 'custom_analysis.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")

    plt.show()

# Then call in main():
plotter.plot_custom_analysis(df)
```

---

## Understanding the Code Logic

### Why Cumulative Sum for Tunnel Length?

```python
tunnel_length = np.cumsum(np.random.uniform(0.5, 2.0, n_samples))
```

**Reason**: Tunnel length only increases (never decreases)
- `random.uniform(0.5, 2.0)` = progress per interval (0.5-2.0 meters)
- `cumsum()` = adds each value to previous total
- Result: Realistic increasing tunnel length

### Why Random Walk for Deviations?

```python
hor_deviation_machine = np.cumsum(np.random.normal(0, 2, n_samples))
```

**Reason**: Deviations drift over time and require correction
- `random.normal(0, 2)` = small random steps (mean=0, std=2mm)
- `cumsum()` = each step builds on previous position
- Result: Realistic wandering that requires steering correction

### Why Mirror Cylinder Behavior?

```python
# Opposite cylinders should work against each other
cylinder_01 = base_stroke + offset_1
cylinder_03 = base_stroke - offset_1  # Opposite of cylinder_01
```

**Reason**: Physical layout of steering system
- Cylinders 1 & 3 are opposite sides (left-right)
- When left extends, right retracts → machine steers right
- Creates realistic coordinated steering

### Why Calculate Derived Metrics?

```python
df['drilling_efficiency'] = df['advance_speed_mm_min'] / df['working_pressure_bar']
```

**Reason**: Single metric for complex relationship
- Raw values: Speed=30mm/min, Pressure=150bar
- Efficiency: 0.20 mm/min/bar
- Easy to spot degradation: Efficiency dropping from 0.20 to 0.15
- Accounts for both factors in one number

---

## Common Modifications

### 1. Change Output Directory

```python
# In __init__(), modify:
self.base_dir = Path("C:/My Custom Path/MTBM Analysis")
```

### 2. Generate More/Less Data

```python
# In main(), change:
df = plotter.generate_synthetic_mtbm_data(n_samples=5000)  # Was 1000
```

### 3. Disable Plot Display (Server Mode)

```python
# In each plot function, remove or comment out:
# plt.show()

# Or add at top of script:
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
```

### 4. Add Timestamp to Filenames

```python
# In plot functions:
from datetime import datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_path = self.plots_dir / f'mtbm_time_series_{timestamp}.png'
```

### 5. Export to Different Format

```python
# Change from PNG to PDF:
plt.savefig(save_path.with_suffix('.pdf'), format='pdf', dpi=300)

# Or SVG (vector graphics):
plt.savefig(save_path.with_suffix('.svg'), format='svg')
```

---

## Debugging Tips

### Problem: Plots not saving

**Check**:
```python
print(f"Saving to: {save_path}")
print(f"Directory exists: {save_path.parent.exists()}")
print(f"Can write: {os.access(save_path.parent, os.W_OK)}")
```

### Problem: Wrong directory structure

**Check**:
```python
print(f"Base dir: {self.base_dir}")
print(f"Plots dir: {self.plots_dir}")
print(f"Script location: {Path(__file__).parent}")
```

### Problem: Data looks wrong

**Check**:
```python
print(df.head())  # First 5 rows
print(df.describe())  # Statistics for all columns
print(df.dtypes)  # Data types
print(df.isnull().sum())  # Missing values
```

### Problem: Plots too crowded/sparse

**Adjust**:
```python
# More spacing
plt.tight_layout(pad=3.0)  # Increase padding

# Larger figure
fig, axes = plt.subplots(6, 4, figsize=(25, 30))  # Increase size

# Fewer grid lines
ax.grid(True, alpha=0.1)  # More transparent
```

---

## Performance Considerations

### Large Datasets

For >10,000 rows:

```python
# Use downsampling for plots
df_plot = df.iloc[::10]  # Every 10th row for plotting
# Still use full df for calculations

# Or use rolling windows
df_rolling = df.rolling(window=50).mean()  # Smooth data
```

### Memory Usage

```python
# Check memory usage
print(f"DataFrame memory: {df.memory_usage().sum() / 1024**2:.2f} MB")

# Optimize data types
df['survey_mode'] = df['survey_mode'].astype('int8')  # Was int64
df['activated_interjack'] = df['activated_interjack'].astype('int8')
```

### Faster Execution

```python
# Generate plots in parallel (advanced)
from multiprocessing import Pool

def plot_wrapper(args):
    plot_func, df = args
    plot_func(df)

with Pool(4) as p:  # 4 parallel processes
    p.map(plot_wrapper, [
        (plotter.plot_time_series_overview, df),
        (plotter.plot_deviation_analysis, df),
        # etc.
    ])
```

---

## Next Steps

1. **Learn by Doing**: Run the script and examine each generated plot
2. **Modify Gradually**: Change one parameter at a time
3. **Compare Results**: See how changes affect output
4. **Use Real Data**: Apply to actual MTBM datasets
5. **Extend Framework**: Add domain-specific analyses

---

**Document Version**: 1.0
**Last Updated**: November 2024
**Maintainer**: MTBM ML Framework Team
