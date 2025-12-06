#!/usr/bin/env python3
"""
MTBM Comprehensive Plotting Framework
====================================

Professional visualization system for all 23 key MTBM operational parameters:
1. Time series analysis for each parameter
2. Cross-parameter correlation analysis
3. Performance trend identification
4. Operational efficiency monitoring
5. Quality control charts

Author: MTBM ML Framework
Date: November 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import os
from pathlib import Path
warnings.filterwarnings('ignore')

class MTBMComprehensivePlotter:
    """
    Comprehensive plotting system for MTBM operational data
    
    Handles all 23 key parameters with professional visualizations:
    - Time series plots
    - Correlation analysis
    - Performance monitoring
    - Quality control charts
    - Operational efficiency analysis
    """
    
    def __init__(self, base_dir=None):
        # Set up directory structure
        if base_dir is None:
            base_dir = Path(__file__).parent.parent
        else:
            base_dir = Path(base_dir)

        self.base_dir = base_dir
        self.output_dir = base_dir / 'outputs'
        self.plots_dir = self.output_dir / 'plots'
        self.data_dir = base_dir / 'data'
        self.processed_data_dir = self.data_dir / 'processed'
        self.reports_dir = self.output_dir / 'reports'

        # Create directories if they don't exist
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

        # Define all 23 MTBM parameters
        self.parameters = {
            'temporal': {
                'date': 'Date',
                'time': 'Time'
            },
            'survey_position': {
                'tunnel_length_m': 'Survey: Tunnel length, m',
                'hor_deviation_machine_mm': 'Survey: Hor. deviation machine, mm',
                'vert_deviation_machine_mm': 'Survey: Vert. deviation machine, mm',
                'hor_deviation_drill_head_mm': 'Survey: Hor. deviation drill head tip, mm',
                'vert_deviation_drill_head_mm': 'Survey: Vert. deviation drill head tip, mm'
            },
            'survey_orientation': {
                'yaw_mm_per_m': 'Survey: Yaw, mm/m',
                'pitch_mm_per_m': 'Survey: Pitch, mm/m',
                'reel_degree': 'Survey: Reel, Degree',
                'temperature_els_mwd': 'Survey: Temperature in ELS/MWD, Degree',
                'survey_mode': 'Survey: Mode (0=ELS,1=ELS-HWL,2=GNS)'
            },
            'steering_control': {
                'cylinder_01_stroke_mm': 'SC: Cylinder 01 stroke, mm',
                'cylinder_02_stroke_mm': 'SC: Cylinder 02 stroke, mm',
                'cylinder_03_stroke_mm': 'SC: Cylinder 03 stroke, mm',
                'cylinder_04_stroke_mm': 'SC: Cylinder 04 stroke, mm',
                'total_force_kn': 'SC: Total force, kN'
            },
            'operational': {
                'advance_speed_mm_min': 'Survey: Advance speed, mm/min',
                'interjack_force_kn': 'Interjack: Force of TC and interjack, kN',
                'activated_interjack': 'Interjack: Currently activated interjack'
            },
            'cutter_wheel': {
                'working_pressure_bar': 'CW: Working pressure, bar',
                'revolution_rpm': 'CW: Revolution, rpm',
                'earth_pressure_01_bar': 'CW: Earth pressure 01 of excavation chamber, bar'
            }
        }
        
        # Color schemes for different parameter groups
        self.color_schemes = {
            'temporal': '#1f77b4',
            'survey_position': '#ff7f0e',
            'survey_orientation': '#2ca02c',
            'steering_control': '#d62728',
            'operational': '#9467bd',
            'cutter_wheel': '#8c564b'
        }
        
        # Set professional plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
    def generate_synthetic_mtbm_data(self, n_samples=1000):
        """
        Generate comprehensive synthetic MTBM data with all 23 parameters
        """
        print("Generating comprehensive MTBM operational data...")
        
        np.random.seed(42)
        
        # Generate base time series
        start_date = datetime(2024, 1, 1, 8, 0, 0)  # Start at 8 AM
        timestamps = []
        dates = []
        times = []
        
        for i in range(n_samples):
            current_time = start_date + timedelta(minutes=i*30)  # 30-minute intervals
            timestamps.append(current_time)
            dates.append(current_time.date())
            times.append(current_time.time())
        
        # Tunnel progress (cumulative)
        tunnel_length = np.cumsum(np.random.uniform(0.5, 2.0, n_samples))  # 0.5-2m per 30min
        
        # Survey deviations (with drift and corrections)
        hor_deviation_machine = np.cumsum(np.random.normal(0, 2, n_samples))
        vert_deviation_machine = np.cumsum(np.random.normal(0, 1.5, n_samples))
        
        # Drill head deviations (slightly different from machine)
        hor_deviation_drill_head = hor_deviation_machine + np.random.normal(0, 5, n_samples)
        vert_deviation_drill_head = vert_deviation_machine + np.random.normal(0, 3, n_samples)
        
        # Survey orientation parameters
        yaw = np.random.normal(0, 2, n_samples)  # mm/m
        pitch = np.random.normal(0, 1.5, n_samples)  # mm/m
        reel = np.random.uniform(-180, 180, n_samples)  # degrees
        temperature_els = np.random.normal(25, 5, n_samples)  # Celsius
        survey_mode = np.random.choice([0, 1, 2], n_samples, p=[0.6, 0.3, 0.1])
        
        # Steering control cylinders (coordinated movement)
        base_stroke = 50  # mm base position
        cylinder_01 = base_stroke + np.random.normal(0, 20, n_samples)
        cylinder_02 = base_stroke + np.random.normal(0, 20, n_samples)
        cylinder_03 = base_stroke + np.random.normal(0, 20, n_samples)
        cylinder_04 = base_stroke + np.random.normal(0, 20, n_samples)
        
        # Total steering force (related to cylinder positions)
        total_force = 500 + np.abs(cylinder_01 - base_stroke) * 2 + \
                     np.abs(cylinder_02 - base_stroke) * 2 + \
                     np.abs(cylinder_03 - base_stroke) * 2 + \
                     np.abs(cylinder_04 - base_stroke) * 2 + \
                     np.random.normal(0, 50, n_samples)
        
        # Operational parameters
        advance_speed = np.random.uniform(15, 45, n_samples)  # mm/min
        interjack_force = np.random.uniform(800, 1500, n_samples)  # kN
        activated_interjack = np.random.randint(1, 5, n_samples)
        
        # Cutter wheel parameters (correlated)
        working_pressure = np.random.uniform(120, 200, n_samples)  # bar
        revolution_rpm = np.random.uniform(6, 12, n_samples)
        earth_pressure = working_pressure * np.random.uniform(0.7, 1.3, n_samples)  # Related to working pressure
        
        # Create comprehensive DataFrame
        data = {
            # Temporal
            'timestamp': timestamps,
            'date': dates,
            'time': times,
            
            # Survey Position
            'tunnel_length_m': tunnel_length,
            'hor_deviation_machine_mm': hor_deviation_machine,
            'vert_deviation_machine_mm': vert_deviation_machine,
            'hor_deviation_drill_head_mm': hor_deviation_drill_head,
            'vert_deviation_drill_head_mm': vert_deviation_drill_head,
            
            # Survey Orientation
            'yaw_mm_per_m': yaw,
            'pitch_mm_per_m': pitch,
            'reel_degree': reel,
            'temperature_els_mwd': temperature_els,
            'survey_mode': survey_mode,
            
            # Steering Control
            'cylinder_01_stroke_mm': cylinder_01,
            'cylinder_02_stroke_mm': cylinder_02,
            'cylinder_03_stroke_mm': cylinder_03,
            'cylinder_04_stroke_mm': cylinder_04,
            'total_force_kn': total_force,
            
            # Operational
            'advance_speed_mm_min': advance_speed,
            'interjack_force_kn': interjack_force,
            'activated_interjack': activated_interjack,
            
            # Cutter Wheel
            'working_pressure_bar': working_pressure,
            'revolution_rpm': revolution_rpm,
            'earth_pressure_01_bar': earth_pressure
        }
        
        df = pd.DataFrame(data)
        
        # Add calculated parameters
        df['total_deviation_mm'] = np.sqrt(df['hor_deviation_machine_mm']**2 + 
                                          df['vert_deviation_machine_mm']**2)
        df['drilling_efficiency'] = df['advance_speed_mm_min'] / df['working_pressure_bar']
        df['power_efficiency'] = df['advance_speed_mm_min'] / df['revolution_rpm']
        
        return df
    
    def plot_time_series_overview(self, df, save_plots=True):
        """
        Create comprehensive time series overview of all parameters
        """
        print("Creating time series overview plots...")
        
        # Create large subplot grid
        fig, axes = plt.subplots(6, 4, figsize=(20, 24))
        fig.suptitle('MTBM Comprehensive Operational Parameters - Time Series Analysis', 
                     fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes_flat = axes.flatten()
        
        # Plot each parameter group
        plot_idx = 0
        
        # 1. Survey Position Parameters
        position_params = ['tunnel_length_m', 'hor_deviation_machine_mm', 
                          'vert_deviation_machine_mm', 'total_deviation_mm']
        for param in position_params:
            if param in df.columns:
                axes_flat[plot_idx].plot(df['timestamp'], df[param], 
                                       color=self.color_schemes['survey_position'], linewidth=1)
                axes_flat[plot_idx].set_title(f'{param.replace("_", " ").title()}')
                axes_flat[plot_idx].tick_params(axis='x', rotation=45)
                axes_flat[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # 2. Survey Orientation Parameters
        orientation_params = ['yaw_mm_per_m', 'pitch_mm_per_m', 'reel_degree', 'temperature_els_mwd']
        for param in orientation_params:
            if param in df.columns:
                axes_flat[plot_idx].plot(df['timestamp'], df[param], 
                                       color=self.color_schemes['survey_orientation'], linewidth=1)
                axes_flat[plot_idx].set_title(f'{param.replace("_", " ").title()}')
                axes_flat[plot_idx].tick_params(axis='x', rotation=45)
                axes_flat[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # 3. Steering Control Parameters
        steering_params = ['cylinder_01_stroke_mm', 'cylinder_02_stroke_mm', 
                          'cylinder_03_stroke_mm', 'cylinder_04_stroke_mm']
        for param in steering_params:
            if param in df.columns:
                axes_flat[plot_idx].plot(df['timestamp'], df[param], 
                                       color=self.color_schemes['steering_control'], linewidth=1)
                axes_flat[plot_idx].set_title(f'{param.replace("_", " ").title()}')
                axes_flat[plot_idx].tick_params(axis='x', rotation=45)
                axes_flat[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # 4. Operational Parameters
        operational_params = ['advance_speed_mm_min', 'interjack_force_kn', 
                             'total_force_kn', 'activated_interjack']
        for param in operational_params:
            if param in df.columns:
                axes_flat[plot_idx].plot(df['timestamp'], df[param], 
                                       color=self.color_schemes['operational'], linewidth=1)
                axes_flat[plot_idx].set_title(f'{param.replace("_", " ").title()}')
                axes_flat[plot_idx].tick_params(axis='x', rotation=45)
                axes_flat[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # 5. Cutter Wheel Parameters
        cutter_params = ['working_pressure_bar', 'revolution_rpm', 'earth_pressure_01_bar']
        for param in cutter_params:
            if param in df.columns:
                axes_flat[plot_idx].plot(df['timestamp'], df[param], 
                                       color=self.color_schemes['cutter_wheel'], linewidth=1)
                axes_flat[plot_idx].set_title(f'{param.replace("_", " ").title()}')
                axes_flat[plot_idx].tick_params(axis='x', rotation=45)
                axes_flat[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # 6. Efficiency Parameters
        efficiency_params = ['drilling_efficiency', 'power_efficiency']
        for param in efficiency_params:
            if param in df.columns:
                axes_flat[plot_idx].plot(df['timestamp'], df[param], 
                                       color='purple', linewidth=1)
                axes_flat[plot_idx].set_title(f'{param.replace("_", " ").title()}')
                axes_flat[plot_idx].tick_params(axis='x', rotation=45)
                axes_flat[plot_idx].grid(True, alpha=0.3)
                plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, len(axes_flat)):
            axes_flat[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_plots:
            save_path = self.plots_dir / 'mtbm_time_series_overview.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_deviation_analysis(self, df, save_plots=True):
        """
        Specialized plots for tunnel deviation analysis
        """
        print("Creating deviation analysis plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MTBM Tunnel Deviation Analysis', fontsize=16, fontweight='bold')
        
        # 1. Horizontal vs Vertical Deviation Scatter
        axes[0,0].scatter(df['hor_deviation_machine_mm'], df['vert_deviation_machine_mm'], 
                         alpha=0.6, c=df['tunnel_length_m'], cmap='viridis')
        axes[0,0].set_xlabel('Horizontal Deviation (mm)')
        axes[0,0].set_ylabel('Vertical Deviation (mm)')
        axes[0,0].set_title('Machine Deviation Pattern')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add deviation tolerance circles
        circle1 = plt.Circle((0, 0), 25, fill=False, color='green', linestyle='--', label='±25mm tolerance')
        circle2 = plt.Circle((0, 0), 50, fill=False, color='orange', linestyle='--', label='±50mm tolerance')
        circle3 = plt.Circle((0, 0), 75, fill=False, color='red', linestyle='--', label='±75mm tolerance')
        axes[0,0].add_patch(circle1)
        axes[0,0].add_patch(circle2)
        axes[0,0].add_patch(circle3)
        axes[0,0].legend()
        axes[0,0].axis('equal')
        
        # 2. Total Deviation Over Time
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
        axes[1,0].plot(df['timestamp'], df['cylinder_01_stroke_mm'], label='Cylinder 1', alpha=0.7)
        axes[1,0].plot(df['timestamp'], df['cylinder_02_stroke_mm'], label='Cylinder 2', alpha=0.7)
        axes[1,0].plot(df['timestamp'], df['cylinder_03_stroke_mm'], label='Cylinder 3', alpha=0.7)
        axes[1,0].plot(df['timestamp'], df['cylinder_04_stroke_mm'], label='Cylinder 4', alpha=0.7)
        axes[1,0].set_xlabel('Time')
        axes[1,0].set_ylabel('Cylinder Stroke (mm)')
        axes[1,0].set_title('Steering Cylinder Activity')
        axes[1,0].legend()
        axes[1,0].tick_params(axis='x', rotation=45)
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Deviation vs Steering Force
        axes[1,1].scatter(df['total_force_kn'], df['total_deviation_mm'], 
                         alpha=0.6, c=df['advance_speed_mm_min'], cmap='plasma')
        axes[1,1].set_xlabel('Total Steering Force (kN)')
        axes[1,1].set_ylabel('Total Deviation (mm)')
        axes[1,1].set_title('Steering Force vs Deviation')
        cbar = plt.colorbar(axes[1,1].collections[0], ax=axes[1,1])
        cbar.set_label('Advance Speed (mm/min)')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            save_path = self.plots_dir / 'mtbm_deviation_analysis.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_performance_dashboard(self, df, save_plots=True):
        """
        Create operational performance dashboard
        """
        print("Creating performance dashboard...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('MTBM Operational Performance Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Advance Speed vs Working Pressure
        scatter = axes[0,0].scatter(df['working_pressure_bar'], df['advance_speed_mm_min'], 
                                   c=df['revolution_rpm'], cmap='viridis', alpha=0.6)
        axes[0,0].set_xlabel('Working Pressure (bar)')
        axes[0,0].set_ylabel('Advance Speed (mm/min)')
        axes[0,0].set_title('Speed vs Pressure Performance')
        plt.colorbar(scatter, ax=axes[0,0], label='RPM')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Drilling Efficiency Over Time
        axes[0,1].plot(df['tunnel_length_m'], df['drilling_efficiency'], 
                      color='green', linewidth=2)
        axes[0,1].set_xlabel('Tunnel Length (m)')
        axes[0,1].set_ylabel('Drilling Efficiency (mm/min/bar)')
        axes[0,1].set_title('Drilling Efficiency Trend')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Earth Pressure vs Working Pressure
        axes[0,2].scatter(df['working_pressure_bar'], df['earth_pressure_01_bar'], 
                         alpha=0.6, color='brown')
        axes[0,2].plot([df['working_pressure_bar'].min(), df['working_pressure_bar'].max()],
                      [df['working_pressure_bar'].min(), df['working_pressure_bar'].max()],
                      'r--', label='1:1 ratio')
        axes[0,2].set_xlabel('Working Pressure (bar)')
        axes[0,2].set_ylabel('Earth Pressure (bar)')
        axes[0,2].set_title('Pressure Balance Analysis')
        axes[0,2].legend()
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Interjack Force Distribution
        axes[1,0].hist(df['interjack_force_kn'], bins=30, alpha=0.7, color='purple')
        axes[1,0].axvline(df['interjack_force_kn'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["interjack_force_kn"].mean():.1f} kN')
        axes[1,0].set_xlabel('Interjack Force (kN)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Interjack Force Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Temperature Monitoring
        axes[1,1].plot(df['timestamp'], df['temperature_els_mwd'], 
                      color='orange', linewidth=1)
        axes[1,1].axhline(y=30, color='red', linestyle='--', label='High temp warning')
        axes[1,1].axhline(y=15, color='blue', linestyle='--', label='Low temp warning')
        axes[1,1].set_xlabel('Time')
        axes[1,1].set_ylabel('Temperature (°C)')
        axes[1,1].set_title('ELS/MWD Temperature Monitoring')
        axes[1,1].legend()
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Survey Mode Usage
        mode_counts = df['survey_mode'].value_counts()
        mode_labels = ['ELS', 'ELS-HWL', 'GNS']
        axes[1,2].pie(mode_counts.values, labels=[mode_labels[i] for i in mode_counts.index], 
                     autopct='%1.1f%%', startangle=90)
        axes[1,2].set_title('Survey Mode Usage Distribution')
        
        plt.tight_layout()
        
        if save_plots:
            save_path = self.plots_dir / 'mtbm_performance_dashboard.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
    
    def plot_correlation_matrix(self, df, save_plots=True):
        """
        Create correlation matrix for all numerical parameters
        """
        print("Creating correlation analysis...")
        
        # Select numerical columns
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove timestamp-related columns for correlation
        exclude_cols = ['survey_mode', 'activated_interjack']
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        # Calculate correlation matrix
        correlation_matrix = df[numerical_cols].corr()
        
        # Create correlation heatmap
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
        
        sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                   center=0, square=True, linewidths=0.5, cbar_kws={"shrink": .8},
                   fmt='.2f')
        
        plt.title('MTBM Parameters Correlation Matrix', fontsize=16, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        
        if save_plots:
            save_path = self.plots_dir / 'mtbm_correlation_matrix.png'
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        
        # Print top correlations
        print("\nTop 10 Strongest Correlations:")
        print("-" * 50)
        
        # Get correlation pairs
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
        
        # Sort by absolute correlation
        correlation_pairs.sort(key=lambda x: x[2], reverse=True)
        
        for i, (param1, param2, abs_corr, corr) in enumerate(correlation_pairs[:10]):
            print(f"{i+1:2d}. {param1} ↔ {param2}")
            print(f"    Correlation: {corr:+.3f}")
            print()
    
    def generate_comprehensive_report(self, df):
        """
        Generate comprehensive analysis report
        """
        print("\n" + "="*80)
        print("MTBM COMPREHENSIVE OPERATIONAL ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        print(f"\n1. DATA OVERVIEW")
        print(f"   Total Records: {len(df):,}")
        print(f"   Time Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"   Tunnel Length: {df['tunnel_length_m'].max():.1f} meters")
        
        # Performance metrics
        print(f"\n2. PERFORMANCE METRICS")
        print(f"   Average Advance Speed: {df['advance_speed_mm_min'].mean():.1f} mm/min")
        print(f"   Average Working Pressure: {df['working_pressure_bar'].mean():.1f} bar")
        print(f"   Average Revolution: {df['revolution_rpm'].mean():.1f} rpm")
        print(f"   Drilling Efficiency: {df['drilling_efficiency'].mean():.3f} mm/min/bar")
        
        # Deviation analysis
        print(f"\n3. DEVIATION ANALYSIS")
        print(f"   Max Horizontal Deviation: {df['hor_deviation_machine_mm'].max():.1f} mm")
        print(f"   Max Vertical Deviation: {df['vert_deviation_machine_mm'].max():.1f} mm")
        print(f"   Max Total Deviation: {df['total_deviation_mm'].max():.1f} mm")
        print(f"   Average Total Deviation: {df['total_deviation_mm'].mean():.1f} mm")
        
        # Quality assessment
        deviation_quality = []
        for dev in df['total_deviation_mm']:
            if dev <= 25:
                deviation_quality.append('Excellent')
            elif dev <= 50:
                deviation_quality.append('Good')
            elif dev <= 75:
                deviation_quality.append('Acceptable')
            else:
                deviation_quality.append('Poor')
        
        quality_counts = pd.Series(deviation_quality).value_counts(normalize=True)
        print(f"\n4. QUALITY DISTRIBUTION")
        for quality, percentage in quality_counts.items():
            print(f"   {quality}: {percentage:.1%}")
        
        # Operational insights
        print(f"\n5. OPERATIONAL INSIGHTS")
        print(f"   Temperature Range: {df['temperature_els_mwd'].min():.1f}°C to {df['temperature_els_mwd'].max():.1f}°C")
        print(f"   Interjack Force Range: {df['interjack_force_kn'].min():.0f} to {df['interjack_force_kn'].max():.0f} kN")
        print(f"   Total Steering Force Range: {df['total_force_kn'].min():.0f} to {df['total_force_kn'].max():.0f} kN")
        
        print("="*80)

def main():
    """
    Main execution function for MTBM comprehensive plotting
    """
    print("MTBM Comprehensive Plotting Framework")
    print("====================================")
    
    # Initialize plotter
    plotter = MTBMComprehensivePlotter()
    
    # Generate comprehensive synthetic data
    df = plotter.generate_synthetic_mtbm_data(n_samples=1000)
    
    # Save the dataset
    csv_path = plotter.processed_data_dir / 'mtbm_comprehensive_data.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved comprehensive dataset: {csv_path}")
    
    # Create all visualizations
    print("\nGenerating comprehensive visualizations...")
    
    # 1. Time series overview
    plotter.plot_time_series_overview(df)
    
    # 2. Deviation analysis
    plotter.plot_deviation_analysis(df)
    
    # 3. Performance dashboard
    plotter.plot_performance_dashboard(df)
    
    # 4. Correlation analysis
    plotter.plot_correlation_matrix(df)
    
    # 5. Generate comprehensive report
    plotter.generate_comprehensive_report(df)
    
    print("\nAll visualizations completed!")
    print("\nGenerated files:")
    print(f"Data:")
    print(f"  - {plotter.processed_data_dir / 'mtbm_comprehensive_data.csv'}")
    print(f"\nPlots:")
    print(f"  - {plotter.plots_dir / 'mtbm_time_series_overview.png'}")
    print(f"  - {plotter.plots_dir / 'mtbm_deviation_analysis.png'}")
    print(f"  - {plotter.plots_dir / 'mtbm_performance_dashboard.png'}")
    print(f"  - {plotter.plots_dir / 'mtbm_correlation_matrix.png'}")

if __name__ == "__main__":
    main()
