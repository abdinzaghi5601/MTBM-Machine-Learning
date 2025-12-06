# MTBM Deviation Graph Generation Guide

## Overview

This guide shows you how to generate comprehensive deviation graphs for your MTBM data using the visualization tools provided.

## Quick Start (Just Generated!)

✅ **You just generated deviation analysis for your AVN1200 data!**

**Key Results from Your Data**:
- **377 readings** from 556.37m of tunnel
- **Average deviation**: 13.80mm
- **Quality distribution**: 29.4% Excellent, 69.5% Poor readings
- **Alignment**: Good overall component alignment (1.67mm difference)

## Generated Files

1. **`deviation_analysis.csv`** - Complete processed data export
2. **ASCII plots** - Text-based visualizations (shown above)
3. **Statistics report** - Comprehensive analysis

## Available Graph Types

### 1. Basic ASCII Graphs (✅ Just Generated)
```bash
python3 simple_deviation_graphs.py
```

**Outputs**:
- Total deviation over distance (ASCII plot)
- Quality distribution bar chart
- Horizontal vs vertical deviation scatter plot
- Complete statistics report

### 2. Advanced Matplotlib Graphs (Requires Installation)

**Install Requirements**:
```bash
pip install pandas matplotlib seaborn numpy
```

**Generate Advanced Graphs**:
```bash
python3 deviation_visualization.py
```

**Advanced Graph Types**:
- **Deviation over Distance**: Multiple line plots with thresholds
- **Deviation Patterns**: Scatter plots, histograms, box plots
- **Time Series Analysis**: Time-based trend analysis
- **Quality Heatmaps**: Color-coded performance maps

## Graph Types Explained

### 📊 Total Deviation Over Distance
Shows how your tunneling accuracy changes along the tunnel length:
- **Green zone**: <5mm (Excellent)
- **Orange zone**: 5-15mm (Good) 
- **Red zone**: 15-25mm (Acceptable)
- **Dark red**: >25mm (Poor)

**Your Results**: Most readings in poor range (>25mm), suggests need for steering system review.

### 📈 Deviation Trends
Identifies if performance is improving or declining:
- **Positive trend**: Getting worse (need intervention)
- **Negative trend**: Improving (corrections working)
- **Flat trend**: Stable performance

### 🎯 Alignment Quality Score
Normalized 0-1 quality metric:
- **>0.8**: Excellent steering performance
- **0.5-0.8**: Good performance
- **0.3-0.5**: Acceptable, needs regular corrections
- **<0.3**: Poor performance, major intervention needed

**Your Average**: 0.339 (Acceptable range, regular corrections recommended)

### 📋 Quality Distribution
Shows percentage of readings in each quality category:
- **Your Distribution**: 29.4% Excellent, 0% Good, 1.1% Acceptable, 69.5% Poor
- **Recommendation**: High poor percentage indicates systematic steering issues

## Custom Graph Generation

### For Your Specific Data
```python
# Load your AVN1200 data
from simple_deviation_graphs import SimpleDeviationGraphGenerator

analyzer = SimpleDeviationGraphGenerator()
analyzer.load_real_data("AVN1200-ML/measure_protocol_original_.xls.csv")
analyzer.process_deviation_features()
analyzer.run_complete_analysis()
```

### For AVN800 Data
```python
# Load AVN800 data
analyzer = SimpleDeviationGraphGenerator()
analyzer.load_real_data("AVN800-Drive-Protocol/your_data.csv")
analyzer.process_deviation_features()
analyzer.run_complete_analysis()
```

## Understanding Your Results

### 🔴 **Key Findings from Your Data**:

1. **High Deviation Levels**: 
   - Average 13.80mm total deviation
   - Maximum deviation reached 66.29mm
   - 69.5% of readings classified as "Poor"

2. **Component Alignment**: 
   - Good machine vs drill head alignment (1.67mm average difference)
   - No systematic bias between components

3. **Operational Patterns**:
   - Wide horizontal deviation range (-43mm to +65mm) 
   - Vertical deviation more controlled (-32mm to +26mm)
   - Some excellent periods (29.4% excellent readings)

### 📋 **Recommendations Based on Your Data**:

1. **Steering System Review**: High poor percentage suggests systematic issues
2. **Regular Corrections**: Moderate quality score needs consistent adjustments  
3. **Operator Training**: Wide deviation ranges indicate inconsistent operation
4. **Equipment Calibration**: Check steering cylinder calibration and response

## Advanced Analysis Options

### 1. Trend Analysis
```python
# Calculate trends over specific segments
analyzer.process_deviation_features()
# Analyze first 100m vs last 100m performance
```

### 2. Quality Threshold Adjustment
```python
# Customize thresholds for your specific project
# Modify quality categories based on project tolerances
```

### 3. Time-Based Analysis
```python
# Analyze performance by time of day, operator shift, etc.
# Requires datetime information in your data
```

## Export Options

### 1. CSV Export (✅ Already Generated)
- Complete processed data with all calculated features
- Ready for Excel analysis or custom plotting
- File: `deviation_analysis.csv`

### 2. Image Export (With Matplotlib)
```python
# Save high-resolution plots
visualizer.generate_all_deviation_plots("my_deviation_plots/")
```

### 3. Report Generation
```python
# Generate PDF reports with all graphs and statistics
# Custom reporting for project documentation
```

## Next Steps

### Immediate Actions:
1. **Review the generated `deviation_analysis.csv`** for detailed insights
2. **Install matplotlib** for advanced graphical plots:
   ```bash
   pip install pandas matplotlib seaborn numpy
   python3 deviation_visualization.py
   ```

### Operational Improvements:
1. **Address high poor percentage** (69.5%) through systematic review
2. **Implement regular quality monitoring** using alignment quality scores
3. **Train operators** on maintaining <15mm total deviation targets
4. **Consider equipment maintenance** if patterns persist

### Long-term Monitoring:
1. **Set up automated analysis** for real-time monitoring
2. **Establish quality thresholds** specific to your project requirements
3. **Implement trend alerts** for early problem detection

## Technical Support

For questions about graph generation or data analysis:
1. Check the generated statistics report above
2. Review the `deviation_analysis.csv` file for detailed data
3. Use the advanced visualization tools for deeper insights
4. Refer to the ML framework documentation for integration options

---

**Your deviation analysis is complete!** The tools are ready to help you maintain better tunneling accuracy and identify improvement opportunities.