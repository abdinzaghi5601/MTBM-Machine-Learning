#!/usr/bin/env python3
"""
Direct MTBM Graph Creator
========================
Creates all 4 visualization sets directly with error handling
"""

import os
import sys

# Set matplotlib backend before importing pyplot
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import seaborn as sns

def create_sample_data():
    """Create sample MTBM data"""
    print("Creating sample MTBM data...")
    
    np.random.seed(42)
    n = 200
    
    # Time series
    start_time = datetime(2024, 1, 1, 8, 0)
    times = [start_time + timedelta(hours=i) for i in range(n)]
    
    # Generate data for all 23 parameters
    data = {
        'timestamp': times,
        'tunnel_length_m': np.cumsum(np.random.uniform(0.5, 2.0, n)),
        'hor_deviation_machine_mm': np.cumsum(np.random.normal(0, 2, n)),
        'vert_deviation_machine_mm': np.cumsum(np.random.normal(0, 1.5, n)),
        'advance_speed_mm_min': np.random.uniform(15, 45, n),
        'working_pressure_bar': np.random.uniform(120, 200, n),
        'revolution_rpm': np.random.uniform(6, 12, n),
        'earth_pressure_01_bar': np.random.uniform(100, 180, n),
        'temperature_els_mwd': np.random.normal(25, 5, n),
        'interjack_force_kn': np.random.uniform(800, 1500, n),
        'total_force_kn': np.random.uniform(500, 1000, n),
        'cylinder_01_stroke_mm': 50 + np.random.normal(0, 20, n),
        'cylinder_02_stroke_mm': 50 + np.random.normal(0, 20, n),
        'cylinder_03_stroke_mm': 50 + np.random.normal(0, 20, n),
        'cylinder_04_stroke_mm': 50 + np.random.normal(0, 20, n),
        'survey_mode': np.random.choice([0, 1, 2], n, p=[0.6, 0.3, 0.1]),
        'yaw_mm_per_m': np.random.normal(0, 2, n),
        'pitch_mm_per_m': np.random.normal(0, 1.5, n),
        'reel_degree': np.random.uniform(-180, 180, n),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate derived parameters
    df['total_deviation_mm'] = np.sqrt(df['hor_deviation_machine_mm']**2 + 
                                      df['vert_deviation_machine_mm']**2)
    df['drilling_efficiency'] = df['advance_speed_mm_min'] / df['working_pressure_bar']
    
    return df

def graph1_time_series(df):
    """Create time series analysis graph"""
    print("Creating Graph 1: Time Series Analysis...")
    
    try:
        fig, axes = plt.subplots(4, 3, figsize=(18, 16))
        fig.suptitle('MTBM Time Series Analysis - Key Parameters', fontsize=16, fontweight='bold')
        
        # Define key parameters to plot
        params = [
            ('tunnel_length_m', 'Tunnel Length (m)', 'blue'),
            ('advance_speed_mm_min', 'Advance Speed (mm/min)', 'green'),
            ('working_pressure_bar', 'Working Pressure (bar)', 'red'),
            ('hor_deviation_machine_mm', 'Horizontal Deviation (mm)', 'orange'),
            ('vert_deviation_machine_mm', 'Vertical Deviation (mm)', 'purple'),
            ('total_deviation_mm', 'Total Deviation (mm)', 'brown'),
            ('revolution_rpm', 'Revolution (RPM)', 'pink'),
            ('earth_pressure_01_bar', 'Earth Pressure (bar)', 'gray'),
            ('temperature_els_mwd', 'Temperature (°C)', 'cyan'),
            ('interjack_force_kn', 'Interjack Force (kN)', 'darkred'),
            ('drilling_efficiency', 'Drilling Efficiency', 'gold'),
            ('total_force_kn', 'Total Force (kN)', 'navy')
        ]
        
        for i, (param, title, color) in enumerate(params):
            row, col = i // 3, i % 3
            axes[row, col].plot(df['timestamp'], df[param], color=color, linewidth=1.5)
            axes[row, col].set_title(title, fontsize=10)
            axes[row, col].tick_params(axis='x', rotation=45, labelsize=8)
            axes[row, col].grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    # Save to current directory with full path
    import os
    save_path = os.path.join(os.getcwd(), '1_mtbm_time_series.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating time series graph: {e}")
        return False

def graph2_deviation_analysis(df):
    """Create deviation analysis graph"""
    print("Creating Graph 2: Deviation Analysis...")
    
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MTBM Tunnel Deviation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Deviation scatter plot
        axes[0,0].scatter(df['hor_deviation_machine_mm'], df['vert_deviation_machine_mm'], 
                         alpha=0.6, c='blue', s=20)
        axes[0,0].set_xlabel('Horizontal Deviation (mm)')
        axes[0,0].set_ylabel('Vertical Deviation (mm)')
        axes[0,0].set_title('Deviation Pattern')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add tolerance circles
        for radius, color, label in [(25, 'green', '±25mm'), (50, 'orange', '±50mm'), (75, 'red', '±75mm')]:
            circle = plt.Circle((0, 0), radius, fill=False, color=color, linestyle='--', linewidth=2)
            axes[0,0].add_patch(circle)
        axes[0,0].legend(['Data Points', '±25mm tolerance', '±50mm tolerance', '±75mm tolerance'])
        axes[0,0].axis('equal')
        
        # 2. Total deviation trend
        axes[0,1].plot(df['tunnel_length_m'], df['total_deviation_mm'], color='red', linewidth=2)
        axes[0,1].axhline(y=25, color='green', linestyle='--', label='Good (±25mm)')
        axes[0,1].axhline(y=50, color='orange', linestyle='--', label='Acceptable (±50mm)')
        axes[0,1].axhline(y=75, color='red', linestyle='--', label='Poor (±75mm)')
        axes[0,1].set_xlabel('Tunnel Length (m)')
        axes[0,1].set_ylabel('Total Deviation (mm)')
        axes[0,1].set_title('Deviation vs Tunnel Progress')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Steering cylinders
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
        
        # 4. Deviation histogram
        axes[1,1].hist(df['total_deviation_mm'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1,1].axvline(df['total_deviation_mm'].mean(), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {df["total_deviation_mm"].mean():.1f}mm')
        axes[1,1].set_xlabel('Total Deviation (mm)')
        axes[1,1].set_ylabel('Frequency')
        axes[1,1].set_title('Deviation Distribution')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
    plt.tight_layout()
    
    # Save to current directory with full path
    save_path = os.path.join(os.getcwd(), '2_mtbm_deviation_analysis.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating deviation analysis: {e}")
        return False

def graph3_performance_dashboard(df):
    """Create performance dashboard"""
    print("Creating Graph 3: Performance Dashboard...")
    
    try:
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
        
        # 2. Drilling efficiency
        axes[0,1].plot(df['tunnel_length_m'], df['drilling_efficiency'], color='green', linewidth=2)
        axes[0,1].set_xlabel('Tunnel Length (m)')
        axes[0,1].set_ylabel('Drilling Efficiency (mm/min/bar)')
        axes[0,1].set_title('Drilling Efficiency Trend')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Pressure balance
        axes[0,2].scatter(df['working_pressure_bar'], df['earth_pressure_01_bar'], alpha=0.6, color='brown')
        min_p = min(df['working_pressure_bar'].min(), df['earth_pressure_01_bar'].min())
        max_p = max(df['working_pressure_bar'].max(), df['earth_pressure_01_bar'].max())
        axes[0,2].plot([min_p, max_p], [min_p, max_p], 'r--', linewidth=2, label='1:1 ratio')
        axes[0,2].set_xlabel('Working Pressure (bar)')
        axes[0,2].set_ylabel('Earth Pressure (bar)')
        axes[0,2].set_title('Pressure Balance Analysis')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Force distribution
        axes[1,0].hist(df['interjack_force_kn'], bins=20, alpha=0.7, color='purple', edgecolor='black')
        axes[1,0].axvline(df['interjack_force_kn'].mean(), color='red', linestyle='--', linewidth=2,
                         label=f'Mean: {df["interjack_force_kn"].mean():.0f} kN')
        axes[1,0].set_xlabel('Interjack Force (kN)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Interjack Force Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Temperature monitoring
        axes[1,1].plot(df['timestamp'], df['temperature_els_mwd'], color='orange', linewidth=1)
        axes[1,1].axhline(y=30, color='red', linestyle='--', linewidth=2, label='High temp warning')
        axes[1,1].axhline(y=15, color='blue', linestyle='--', linewidth=2, label='Low temp warning')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Temperature (°C)')
        axes[1,1].set_title('Temperature Monitoring')
        axes[1,1].legend()
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Survey mode pie chart
        mode_counts = df['survey_mode'].value_counts()
        mode_labels = ['ELS', 'ELS-HWL', 'GNS']
        colors = ['lightblue', 'lightgreen', 'lightcoral']
        axes[1,2].pie(mode_counts.values, labels=[mode_labels[i] for i in mode_counts.index], 
                     autopct='%1.1f%%', startangle=90, colors=colors)
        axes[1,2].set_title('Survey Mode Usage')
        
    plt.tight_layout()
    
    # Save to current directory with full path
    save_path = os.path.join(os.getcwd(), '3_mtbm_performance_dashboard.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating performance dashboard: {e}")
        return False

def graph4_correlation_matrix(df):
    """Create correlation matrix"""
    print("Creating Graph 4: Correlation Matrix...")
    
    try:
        # Select numerical columns
        numerical_cols = [
            'tunnel_length_m', 'hor_deviation_machine_mm', 'vert_deviation_machine_mm',
            'advance_speed_mm_min', 'working_pressure_bar', 'revolution_rpm',
            'earth_pressure_01_bar', 'temperature_els_mwd', 'interjack_force_kn',
            'total_force_kn', 'total_deviation_mm', 'drilling_efficiency'
        ]
        
        # Calculate correlation matrix
        correlation_matrix = df[numerical_cols].corr()
        
        # Create heatmap
        plt.figure(figsize=(14, 12))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.2f', annot_kws={'size': 9})
        
        plt.title('MTBM Parameters Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save to current directory with full path
    save_path = os.path.join(os.getcwd(), '4_mtbm_correlation_matrix.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating correlation matrix: {e}")
        return False

def main():
    """Main execution"""
    print("🚀 MTBM DIRECT GRAPH GENERATOR")
    print("==============================")
    
    # Create sample data
    df = create_sample_data()
    
    # Save data
    df.to_csv('mtbm_data.csv', index=False)
    print("✅ Saved data: mtbm_data.csv")
    
    # Create all graphs
    success_count = 0
    
    if graph1_time_series(df):
        success_count += 1
    
    if graph2_deviation_analysis(df):
        success_count += 1
        
    if graph3_performance_dashboard(df):
        success_count += 1
        
    if graph4_correlation_matrix(df):
        success_count += 1
    
    print(f"\n🎉 COMPLETED: {success_count}/4 graphs generated successfully!")
    
    if success_count == 4:
        print("\n📁 ALL GRAPH FILES CREATED:")
        print("   1. 1_mtbm_time_series.png")
        print("   2. 2_mtbm_deviation_analysis.png") 
        print("   3. 3_mtbm_performance_dashboard.png")
        print("   4. 4_mtbm_correlation_matrix.png")
        print("   5. mtbm_data.csv")
        
        print("\n💡 TO VIEW THE GRAPHS:")
        print("   - Navigate to your MTBM-Machine-Learning folder")
        print("   - Double-click any PNG file to open in image viewer")
        print("   - Files are high resolution (300 DPI) for presentations")
    else:
        print(f"\n⚠️  Only {success_count} out of 4 graphs were created successfully")

if __name__ == "__main__":
    main()
