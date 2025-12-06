#!/usr/bin/env python3
"""
MTBM Graph Generator - Simplified Version
========================================

This script generates all the MTBM visualization graphs you requested:
1. Time Series Analysis (24 parameters)
2. Deviation Analysis (4 specialized plots)
3. Performance Dashboard (6 KPI plots)
4. Correlation Matrix (parameter relationships)

Author: MTBM ML Framework
Date: November 2024
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for server environments
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('default')
sns.set_palette("husl")

def generate_mtbm_data(n_samples=500):
    """Generate comprehensive MTBM data"""
    print("Generating MTBM operational data...")
    
    np.random.seed(42)
    
    # Generate timestamps
    start_date = datetime(2024, 1, 1, 8, 0, 0)
    timestamps = [start_date + timedelta(minutes=i*30) for i in range(n_samples)]
    
    # Generate all 23 parameters
    data = {
        'timestamp': timestamps,
        'date': [ts.date() for ts in timestamps],
        'time': [ts.time() for ts in timestamps],
        
        # Survey Position (3-7)
        'tunnel_length_m': np.cumsum(np.random.uniform(0.5, 2.0, n_samples)),
        'hor_deviation_machine_mm': np.cumsum(np.random.normal(0, 2, n_samples)),
        'vert_deviation_machine_mm': np.cumsum(np.random.normal(0, 1.5, n_samples)),
        'hor_deviation_drill_head_mm': np.cumsum(np.random.normal(0, 2.5, n_samples)),
        'vert_deviation_drill_head_mm': np.cumsum(np.random.normal(0, 2, n_samples)),
        
        # Survey Orientation (8-12)
        'yaw_mm_per_m': np.random.normal(0, 2, n_samples),
        'pitch_mm_per_m': np.random.normal(0, 1.5, n_samples),
        'reel_degree': np.random.uniform(-180, 180, n_samples),
        'temperature_els_mwd': np.random.normal(25, 5, n_samples),
        'survey_mode': np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1]),
        
        # Steering Control (13-16, 23)
        'cylinder_01_stroke_mm': 50 + np.random.normal(0, 20, n_samples),
        'cylinder_02_stroke_mm': 50 + np.random.normal(0, 20, n_samples),
        'cylinder_03_stroke_mm': 50 + np.random.normal(0, 20, n_samples),
        'cylinder_04_stroke_mm': 50 + np.random.normal(0, 20, n_samples),
        'total_force_kn': np.random.uniform(500, 1000, n_samples),
        
        # Operational (17-19)
        'advance_speed_mm_min': np.random.uniform(15, 45, n_samples),
        'interjack_force_kn': np.random.uniform(800, 1500, n_samples),
        'activated_interjack': np.random.randint(1, 5, n_samples),
        
        # Cutter Wheel (20-22)
        'working_pressure_bar': np.random.uniform(120, 200, n_samples),
        'revolution_rpm': np.random.uniform(6, 12, n_samples),
        'earth_pressure_01_bar': np.random.uniform(100, 180, n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived parameters
    df['total_deviation_mm'] = np.sqrt(df['hor_deviation_machine_mm']**2 + 
                                      df['vert_deviation_machine_mm']**2)
    df['drilling_efficiency'] = df['advance_speed_mm_min'] / df['working_pressure_bar']
    df['power_efficiency'] = df['advance_speed_mm_min'] / df['revolution_rpm']
    
    return df

def create_time_series_plots(df):
    """Create comprehensive time series analysis"""
    print("Creating Time Series Analysis plots...")
    
    fig, axes = plt.subplots(6, 4, figsize=(20, 24))
    fig.suptitle('MTBM Comprehensive Time Series Analysis - All 23 Parameters', 
                 fontsize=16, fontweight='bold')
    
    # Define parameters to plot
    params_to_plot = [
        ('tunnel_length_m', 'Tunnel Length (m)', 'blue'),
        ('hor_deviation_machine_mm', 'Horizontal Deviation (mm)', 'red'),
        ('vert_deviation_machine_mm', 'Vertical Deviation (mm)', 'green'),
        ('total_deviation_mm', 'Total Deviation (mm)', 'purple'),
        ('yaw_mm_per_m', 'Yaw (mm/m)', 'orange'),
        ('pitch_mm_per_m', 'Pitch (mm/m)', 'brown'),
        ('reel_degree', 'Reel (degrees)', 'pink'),
        ('temperature_els_mwd', 'Temperature (°C)', 'cyan'),
        ('cylinder_01_stroke_mm', 'Cylinder 1 Stroke (mm)', 'red'),
        ('cylinder_02_stroke_mm', 'Cylinder 2 Stroke (mm)', 'green'),
        ('cylinder_03_stroke_mm', 'Cylinder 3 Stroke (mm)', 'blue'),
        ('cylinder_04_stroke_mm', 'Cylinder 4 Stroke (mm)', 'orange'),
        ('advance_speed_mm_min', 'Advance Speed (mm/min)', 'darkgreen'),
        ('interjack_force_kn', 'Interjack Force (kN)', 'darkred'),
        ('total_force_kn', 'Total Force (kN)', 'darkblue'),
        ('working_pressure_bar', 'Working Pressure (bar)', 'maroon'),
        ('revolution_rpm', 'Revolution (RPM)', 'navy'),
        ('earth_pressure_01_bar', 'Earth Pressure (bar)', 'darkgray'),
        ('drilling_efficiency', 'Drilling Efficiency', 'gold'),
        ('power_efficiency', 'Power Efficiency', 'silver'),
        ('survey_mode', 'Survey Mode', 'black'),
        ('activated_interjack', 'Active Interjack', 'magenta'),
        ('hor_deviation_drill_head_mm', 'Drill Head H-Dev (mm)', 'lightcoral'),
        ('vert_deviation_drill_head_mm', 'Drill Head V-Dev (mm)', 'lightgreen')
    ]
    
    # Plot each parameter
    for i, (param, title, color) in enumerate(params_to_plot):
        if i < 24:  # Only plot first 24
            row, col = i // 4, i % 4
            if param in df.columns:
                axes[row, col].plot(df['timestamp'], df[param], color=color, linewidth=1, alpha=0.8)
                axes[row, col].set_title(title, fontsize=10)
                axes[row, col].tick_params(axis='x', rotation=45, labelsize=8)
                axes[row, col].tick_params(axis='y', labelsize=8)
                axes[row, col].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mtbm_time_series_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: mtbm_time_series_analysis.png")
    plt.close()

def create_deviation_analysis(df):
    """Create deviation analysis plots"""
    print("Creating Deviation Analysis plots...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('MTBM Tunnel Deviation Analysis', fontsize=16, fontweight='bold')
    
    # 1. Deviation Pattern Scatter
    axes[0,0].scatter(df['hor_deviation_machine_mm'], df['vert_deviation_machine_mm'], 
                     alpha=0.6, c=df['tunnel_length_m'], cmap='viridis', s=20)
    axes[0,0].set_xlabel('Horizontal Deviation (mm)')
    axes[0,0].set_ylabel('Vertical Deviation (mm)')
    axes[0,0].set_title('Machine Deviation Pattern')
    axes[0,0].grid(True, alpha=0.3)
    
    # Add tolerance circles
    for radius, color, label in [(25, 'green', '±25mm'), (50, 'orange', '±50mm'), (75, 'red', '±75mm')]:
        circle = plt.Circle((0, 0), radius, fill=False, color=color, linestyle='--', linewidth=2)
        axes[0,0].add_patch(circle)
    axes[0,0].legend(['±25mm tolerance', '±50mm tolerance', '±75mm tolerance'])
    axes[0,0].axis('equal')
    
    # 2. Total Deviation Trend
    axes[0,1].plot(df['tunnel_length_m'], df['total_deviation_mm'], color='red', linewidth=2)
    axes[0,1].axhline(y=25, color='green', linestyle='--', label='Good (±25mm)')
    axes[0,1].axhline(y=50, color='orange', linestyle='--', label='Acceptable (±50mm)')
    axes[0,1].axhline(y=75, color='red', linestyle='--', label='Poor (±75mm)')
    axes[0,1].set_xlabel('Tunnel Length (m)')
    axes[0,1].set_ylabel('Total Deviation (mm)')
    axes[0,1].set_title('Deviation vs Tunnel Progress')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Steering Cylinder Response
    axes[1,0].plot(df['timestamp'], df['cylinder_01_stroke_mm'], label='Cylinder 1', alpha=0.8)
    axes[1,0].plot(df['timestamp'], df['cylinder_02_stroke_mm'], label='Cylinder 2', alpha=0.8)
    axes[1,0].plot(df['timestamp'], df['cylinder_03_stroke_mm'], label='Cylinder 3', alpha=0.8)
    axes[1,0].plot(df['timestamp'], df['cylinder_04_stroke_mm'], label='Cylinder 4', alpha=0.8)
    axes[1,0].set_xlabel('Time')
    axes[1,0].set_ylabel('Cylinder Stroke (mm)')
    axes[1,0].set_title('Steering Cylinder Coordination')
    axes[1,0].legend()
    axes[1,0].tick_params(axis='x', rotation=45)
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. Quality Statistics
    axes[1,1].hist(df['total_deviation_mm'], bins=30, alpha=0.7, color='purple', edgecolor='black')
    axes[1,1].axvline(df['total_deviation_mm'].mean(), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {df["total_deviation_mm"].mean():.1f}mm')
    axes[1,1].set_xlabel('Total Deviation (mm)')
    axes[1,1].set_ylabel('Frequency')
    axes[1,1].set_title('Deviation Distribution')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mtbm_deviation_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: mtbm_deviation_analysis.png")
    plt.close()

def create_performance_dashboard(df):
    """Create performance dashboard"""
    print("Creating Performance Dashboard...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('MTBM Operational Performance Dashboard', fontsize=16, fontweight='bold')
    
    # 1. Speed vs Pressure
    scatter = axes[0,0].scatter(df['working_pressure_bar'], df['advance_speed_mm_min'], 
                               c=df['revolution_rpm'], cmap='viridis', alpha=0.7, s=30)
    axes[0,0].set_xlabel('Working Pressure (bar)')
    axes[0,0].set_ylabel('Advance Speed (mm/min)')
    axes[0,0].set_title('Speed vs Pressure Performance')
    plt.colorbar(scatter, ax=axes[0,0], label='RPM')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. Drilling Efficiency
    axes[0,1].plot(df['tunnel_length_m'], df['drilling_efficiency'], color='green', linewidth=2)
    axes[0,1].set_xlabel('Tunnel Length (m)')
    axes[0,1].set_ylabel('Drilling Efficiency (mm/min/bar)')
    axes[0,1].set_title('Drilling Efficiency Trend')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. Pressure Balance
    axes[0,2].scatter(df['working_pressure_bar'], df['earth_pressure_01_bar'], alpha=0.6, color='brown')
    min_pressure = min(df['working_pressure_bar'].min(), df['earth_pressure_01_bar'].min())
    max_pressure = max(df['working_pressure_bar'].max(), df['earth_pressure_01_bar'].max())
    axes[0,2].plot([min_pressure, max_pressure], [min_pressure, max_pressure], 'r--', linewidth=2, label='1:1 ratio')
    axes[0,2].set_xlabel('Working Pressure (bar)')
    axes[0,2].set_ylabel('Earth Pressure (bar)')
    axes[0,2].set_title('Pressure Balance Analysis')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. Interjack Force Distribution
    axes[1,0].hist(df['interjack_force_kn'], bins=25, alpha=0.7, color='purple', edgecolor='black')
    axes[1,0].axvline(df['interjack_force_kn'].mean(), color='red', linestyle='--', linewidth=2,
                     label=f'Mean: {df["interjack_force_kn"].mean():.0f} kN')
    axes[1,0].set_xlabel('Interjack Force (kN)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Interjack Force Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. Temperature Monitoring
    axes[1,1].plot(df['timestamp'], df['temperature_els_mwd'], color='orange', linewidth=1)
    axes[1,1].axhline(y=30, color='red', linestyle='--', linewidth=2, label='High temp warning')
    axes[1,1].axhline(y=15, color='blue', linestyle='--', linewidth=2, label='Low temp warning')
    axes[1,1].set_xlabel('Time')
    axes[1,1].set_ylabel('Temperature (°C)')
    axes[1,1].set_title('ELS/MWD Temperature Monitoring')
    axes[1,1].legend()
    axes[1,1].tick_params(axis='x', rotation=45)
    axes[1,1].grid(True, alpha=0.3)
    
    # 6. Survey Mode Usage
    mode_counts = df['survey_mode'].value_counts()
    mode_labels = ['ELS', 'ELS-HWL', 'GNS']
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    axes[1,2].pie(mode_counts.values, labels=[mode_labels[i] for i in mode_counts.index], 
                 autopct='%1.1f%%', startangle=90, colors=colors)
    axes[1,2].set_title('Survey Mode Usage Distribution')
    
    plt.tight_layout()
    plt.savefig('mtbm_performance_dashboard.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: mtbm_performance_dashboard.png")
    plt.close()

def create_correlation_matrix(df):
    """Create correlation matrix"""
    print("Creating Correlation Matrix...")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude_cols = ['survey_mode', 'activated_interjack']
    numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
    
    # Calculate correlation matrix
    correlation_matrix = df[numerical_cols].corr()
    
    # Create heatmap
    plt.figure(figsize=(16, 14))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
               center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
               fmt='.2f', annot_kws={'size': 8})
    
    plt.title('MTBM Parameters Correlation Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig('mtbm_correlation_matrix.png', dpi=300, bbox_inches='tight')
    print("✅ Saved: mtbm_correlation_matrix.png")
    plt.close()
    
    # Print top correlations
    print("\n🔍 TOP 10 STRONGEST CORRELATIONS:")
    print("-" * 50)
    
    correlation_pairs = []
    for i in range(len(correlation_matrix.columns)):
        for j in range(i+1, len(correlation_matrix.columns)):
            corr_value = correlation_matrix.iloc[i, j]
            if not np.isnan(corr_value):
                correlation_pairs.append((
                    correlation_matrix.columns[i],
                    correlation_matrix.columns[j],
                    abs(corr_value),
                    corr_value
                ))
    
    correlation_pairs.sort(key=lambda x: x[2], reverse=True)
    
    for i, (param1, param2, abs_corr, corr) in enumerate(correlation_pairs[:10]):
        print(f"{i+1:2d}. {param1} ↔ {param2}")
        print(f"    Correlation: {corr:+.3f}")

def generate_analysis_report(df):
    """Generate comprehensive analysis report"""
    print("\n" + "="*80)
    print("🚀 MTBM COMPREHENSIVE OPERATIONAL ANALYSIS REPORT")
    print("="*80)
    
    print(f"\n📊 DATA OVERVIEW")
    print(f"   Total Records: {len(df):,}")
    print(f"   Time Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"   Tunnel Length: {df['tunnel_length_m'].max():.1f} meters")
    
    print(f"\n⚡ PERFORMANCE METRICS")
    print(f"   Average Advance Speed: {df['advance_speed_mm_min'].mean():.1f} mm/min")
    print(f"   Average Working Pressure: {df['working_pressure_bar'].mean():.1f} bar")
    print(f"   Average Revolution: {df['revolution_rpm'].mean():.1f} rpm")
    print(f"   Drilling Efficiency: {df['drilling_efficiency'].mean():.3f} mm/min/bar")
    
    print(f"\n📐 DEVIATION ANALYSIS")
    print(f"   Max Horizontal Deviation: {df['hor_deviation_machine_mm'].max():.1f} mm")
    print(f"   Max Vertical Deviation: {df['vert_deviation_machine_mm'].max():.1f} mm")
    print(f"   Max Total Deviation: {df['total_deviation_mm'].max():.1f} mm")
    print(f"   Average Total Deviation: {df['total_deviation_mm'].mean():.1f} mm")
    
    # Quality assessment
    excellent = (df['total_deviation_mm'] <= 25).sum()
    good = ((df['total_deviation_mm'] > 25) & (df['total_deviation_mm'] <= 50)).sum()
    acceptable = ((df['total_deviation_mm'] > 50) & (df['total_deviation_mm'] <= 75)).sum()
    poor = (df['total_deviation_mm'] > 75).sum()
    total = len(df)
    
    print(f"\n🎯 QUALITY DISTRIBUTION")
    print(f"   Excellent (≤25mm): {excellent} ({excellent/total:.1%})")
    print(f"   Good (25-50mm): {good} ({good/total:.1%})")
    print(f"   Acceptable (50-75mm): {acceptable} ({acceptable/total:.1%})")
    print(f"   Poor (>75mm): {poor} ({poor/total:.1%})")
    
    print(f"\n🔧 OPERATIONAL INSIGHTS")
    print(f"   Temperature Range: {df['temperature_els_mwd'].min():.1f}°C to {df['temperature_els_mwd'].max():.1f}°C")
    print(f"   Interjack Force Range: {df['interjack_force_kn'].min():.0f} to {df['interjack_force_kn'].max():.0f} kN")
    print(f"   Total Steering Force Range: {df['total_force_kn'].min():.0f} to {df['total_force_kn'].max():.0f} kN")
    
    print("="*80)

def main():
    """Main execution function"""
    print("🚀 MTBM COMPREHENSIVE GRAPH GENERATOR")
    print("====================================")
    
    # Generate data
    df = generate_mtbm_data(n_samples=500)
    
    # Save dataset
    df.to_csv('mtbm_comprehensive_data.csv', index=False)
    print("✅ Saved dataset: mtbm_comprehensive_data.csv")
    
    # Create all visualizations
    print("\n📊 GENERATING ALL VISUALIZATIONS...")
    
    create_time_series_plots(df)
    create_deviation_analysis(df)
    create_performance_dashboard(df)
    create_correlation_matrix(df)
    
    # Generate report
    generate_analysis_report(df)
    
    print("\n🎉 ALL GRAPHS GENERATED SUCCESSFULLY!")
    print("\n📁 GENERATED FILES:")
    print("   1. mtbm_time_series_analysis.png     - 24 parameter time series")
    print("   2. mtbm_deviation_analysis.png       - Tunnel deviation analysis")
    print("   3. mtbm_performance_dashboard.png    - Operational performance KPIs")
    print("   4. mtbm_correlation_matrix.png       - Parameter correlation heatmap")
    print("   5. mtbm_comprehensive_data.csv       - Complete dataset (500 records)")
    
    print("\n💡 TO VIEW THE GRAPHS:")
    print("   - Open the PNG files in any image viewer")
    print("   - Use Windows Photo Viewer, Paint, or any browser")
    print("   - Files are saved in high resolution (300 DPI)")

if __name__ == "__main__":
    main()
