#!/usr/bin/env python3
"""
Real MTBM Data Plotting Framework
=================================

Specialized plotting system for real MTBM protocol data from Excel/CSV files.
Handles the 23 key parameters with automatic data detection and cleaning.

Usage:
    python plot_real_mtbm_data.py [data_file.csv]
    
If no file specified, will look for common MTBM data files.

Author: MTBM ML Framework
Date: November 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import sys
import os
import warnings
warnings.filterwarnings('ignore')

class RealMTBMDataPlotter:
    """
    Plotting system for real MTBM operational data
    
    Features:
    - Automatic column detection and mapping
    - Data cleaning and validation
    - Professional visualizations
    - Export capabilities
    """
    
    def __init__(self):
        # Common column name variations in MTBM data
        self.column_mappings = {
            'date': ['date', 'Date', 'DATE', 'timestamp', 'Timestamp'],
            'time': ['time', 'Time', 'TIME'],
            'tunnel_length_m': ['tunnel_length', 'length', 'distance', 'chainage', 'station'],
            'hor_deviation_machine_mm': ['hor_deviation', 'horizontal_deviation', 'h_dev', 'x_deviation'],
            'vert_deviation_machine_mm': ['vert_deviation', 'vertical_deviation', 'v_dev', 'y_deviation'],
            'hor_deviation_drill_head_mm': ['hor_dev_head', 'h_dev_tip', 'drill_head_h'],
            'vert_deviation_drill_head_mm': ['vert_dev_head', 'v_dev_tip', 'drill_head_v'],
            'yaw_mm_per_m': ['yaw', 'Yaw', 'YAW'],
            'pitch_mm_per_m': ['pitch', 'Pitch', 'PITCH'],
            'reel_degree': ['reel', 'Reel', 'REEL'],
            'temperature_els_mwd': ['temperature', 'temp', 'Temperature'],
            'survey_mode': ['mode', 'survey_mode', 'Mode'],
            'cylinder_01_stroke_mm': ['cyl1', 'cylinder1', 'stroke1', 'SC1'],
            'cylinder_02_stroke_mm': ['cyl2', 'cylinder2', 'stroke2', 'SC2'],
            'cylinder_03_stroke_mm': ['cyl3', 'cylinder3', 'stroke3', 'SC3'],
            'cylinder_04_stroke_mm': ['cyl4', 'cylinder4', 'stroke4', 'SC4'],
            'advance_speed_mm_min': ['advance_speed', 'speed', 'advance_rate', 'penetration_rate'],
            'interjack_force_kn': ['interjack_force', 'interjack', 'jack_force'],
            'activated_interjack': ['active_jack', 'current_jack', 'jack_number'],
            'working_pressure_bar': ['working_pressure', 'pressure', 'work_pressure'],
            'revolution_rpm': ['revolution', 'rpm', 'RPM', 'rotation'],
            'earth_pressure_01_bar': ['earth_pressure', 'chamber_pressure', 'excavation_pressure'],
            'total_force_kn': ['total_force', 'steering_force', 'thrust_force']
        }
        
        # Set plotting style
        plt.style.use('default')
        sns.set_palette("husl")
    
    def detect_and_map_columns(self, df):
        """
        Automatically detect and map column names to standard format
        """
        print("Detecting and mapping column names...")
        
        mapped_columns = {}
        available_columns = df.columns.tolist()
        
        print(f"Available columns in data: {len(available_columns)}")
        for col in available_columns[:10]:  # Show first 10
            print(f"  - {col}")
        if len(available_columns) > 10:
            print(f"  ... and {len(available_columns) - 10} more")
        
        # Try to map each standard parameter
        for standard_name, possible_names in self.column_mappings.items():
            for possible_name in possible_names:
                # Check for exact match
                if possible_name in available_columns:
                    mapped_columns[standard_name] = possible_name
                    break
                # Check for partial match (case insensitive)
                for col in available_columns:
                    if possible_name.lower() in col.lower() or col.lower() in possible_name.lower():
                        mapped_columns[standard_name] = col
                        break
                if standard_name in mapped_columns:
                    break
        
        print(f"\nMapped {len(mapped_columns)} parameters:")
        for standard, actual in mapped_columns.items():
            print(f"  {standard} → {actual}")
        
        return mapped_columns
    
    def load_and_prepare_data(self, file_path):
        """
        Load and prepare MTBM data from file
        """
        print(f"Loading data from: {file_path}")
        
        # Try to load the file
        try:
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                print("Unsupported file format. Use CSV or Excel files.")
                return None
        except Exception as e:
            print(f"Error loading file: {e}")
            return None
        
        print(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Detect and map columns
        column_mapping = self.detect_and_map_columns(df)
        
        # Create new DataFrame with mapped columns
        mapped_df = pd.DataFrame()
        
        for standard_name, actual_name in column_mapping.items():
            mapped_df[standard_name] = df[actual_name]
        
        # Handle datetime columns
        if 'date' in mapped_df.columns and 'time' in mapped_df.columns:
            try:
                # Combine date and time
                mapped_df['timestamp'] = pd.to_datetime(
                    mapped_df['date'].astype(str) + ' ' + mapped_df['time'].astype(str)
                )
            except:
                print("Could not combine date and time columns")
        elif 'date' in mapped_df.columns:
            try:
                mapped_df['timestamp'] = pd.to_datetime(mapped_df['date'])
            except:
                print("Could not parse date column")
        
        # Calculate derived parameters
        if 'hor_deviation_machine_mm' in mapped_df.columns and 'vert_deviation_machine_mm' in mapped_df.columns:
            mapped_df['total_deviation_mm'] = np.sqrt(
                mapped_df['hor_deviation_machine_mm']**2 + 
                mapped_df['vert_deviation_machine_mm']**2
            )
        
        if 'advance_speed_mm_min' in mapped_df.columns and 'working_pressure_bar' in mapped_df.columns:
            mapped_df['drilling_efficiency'] = (
                mapped_df['advance_speed_mm_min'] / mapped_df['working_pressure_bar']
            )
        
        print(f"Prepared dataset with {len(mapped_df)} records and {len(mapped_df.columns)} parameters")
        
        return mapped_df
    
    def plot_key_parameters(self, df, save_plots=True):
        """
        Plot the most important MTBM parameters
        """
        print("Creating key parameter plots...")
        
        # Determine which plots to create based on available data
        available_params = df.columns.tolist()
        
        # Create subplot grid
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('MTBM Key Operational Parameters', fontsize=16, fontweight='bold')
        
        plot_configs = [
            ('tunnel_length_m', 'Tunnel Length (m)', 'blue'),
            ('advance_speed_mm_min', 'Advance Speed (mm/min)', 'green'),
            ('working_pressure_bar', 'Working Pressure (bar)', 'red'),
            ('hor_deviation_machine_mm', 'Horizontal Deviation (mm)', 'orange'),
            ('vert_deviation_machine_mm', 'Vertical Deviation (mm)', 'purple'),
            ('total_deviation_mm', 'Total Deviation (mm)', 'brown'),
            ('revolution_rpm', 'Revolution (RPM)', 'pink'),
            ('earth_pressure_01_bar', 'Earth Pressure (bar)', 'gray'),
            ('temperature_els_mwd', 'Temperature (°C)', 'cyan')
        ]
        
        for i, (param, title, color) in enumerate(plot_configs):
            row, col = i // 3, i % 3
            
            if param in available_params and not df[param].isna().all():
                if 'timestamp' in df.columns:
                    axes[row, col].plot(df['timestamp'], df[param], color=color, linewidth=1)
                    axes[row, col].tick_params(axis='x', rotation=45)
                else:
                    axes[row, col].plot(df[param], color=color, linewidth=1)
                
                axes[row, col].set_title(title)
                axes[row, col].grid(True, alpha=0.3)
            else:
                axes[row, col].text(0.5, 0.5, f'{title}\n(No Data)', 
                                   ha='center', va='center', transform=axes[row, col].transAxes)
                axes[row, col].set_title(title)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('real_mtbm_key_parameters.png', dpi=300, bbox_inches='tight')
            print("Saved: real_mtbm_key_parameters.png")
        
        plt.show()
    
    def plot_deviation_analysis(self, df, save_plots=True):
        """
        Create deviation analysis plots for real data
        """
        if 'hor_deviation_machine_mm' not in df.columns or 'vert_deviation_machine_mm' not in df.columns:
            print("Deviation data not available for analysis")
            return
        
        print("Creating deviation analysis...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('MTBM Tunnel Deviation Analysis (Real Data)', fontsize=16, fontweight='bold')
        
        # 1. Deviation scatter plot
        axes[0,0].scatter(df['hor_deviation_machine_mm'], df['vert_deviation_machine_mm'], 
                         alpha=0.6, c='blue')
        axes[0,0].set_xlabel('Horizontal Deviation (mm)')
        axes[0,0].set_ylabel('Vertical Deviation (mm)')
        axes[0,0].set_title('Deviation Pattern')
        axes[0,0].grid(True, alpha=0.3)
        
        # Add tolerance circles
        max_dev = max(df['hor_deviation_machine_mm'].abs().max(), 
                     df['vert_deviation_machine_mm'].abs().max())
        for radius, color, label in [(25, 'green', '±25mm'), (50, 'orange', '±50mm'), (75, 'red', '±75mm')]:
            if radius <= max_dev * 1.2:
                circle = plt.Circle((0, 0), radius, fill=False, color=color, linestyle='--', label=label)
                axes[0,0].add_patch(circle)
        axes[0,0].legend()
        axes[0,0].axis('equal')
        
        # 2. Total deviation over time/distance
        if 'total_deviation_mm' in df.columns:
            x_axis = df['tunnel_length_m'] if 'tunnel_length_m' in df.columns else range(len(df))
            x_label = 'Tunnel Length (m)' if 'tunnel_length_m' in df.columns else 'Record Number'
            
            axes[0,1].plot(x_axis, df['total_deviation_mm'], color='red', linewidth=2)
            axes[0,1].axhline(y=25, color='green', linestyle='--', label='Good (±25mm)')
            axes[0,1].axhline(y=50, color='orange', linestyle='--', label='Acceptable (±50mm)')
            axes[0,1].axhline(y=75, color='red', linestyle='--', label='Poor (±75mm)')
            axes[0,1].set_xlabel(x_label)
            axes[0,1].set_ylabel('Total Deviation (mm)')
            axes[0,1].set_title('Deviation Trend')
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Deviation histogram
        axes[1,0].hist(df['total_deviation_mm'], bins=30, alpha=0.7, color='purple')
        axes[1,0].axvline(df['total_deviation_mm'].mean(), color='red', linestyle='--', 
                         label=f'Mean: {df["total_deviation_mm"].mean():.1f}mm')
        axes[1,0].set_xlabel('Total Deviation (mm)')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title('Deviation Distribution')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Deviation statistics
        axes[1,1].axis('off')
        
        # Calculate deviation statistics
        stats_text = f"""
Deviation Statistics:

Mean Total Deviation: {df['total_deviation_mm'].mean():.1f} mm
Max Total Deviation: {df['total_deviation_mm'].max():.1f} mm
Std Deviation: {df['total_deviation_mm'].std():.1f} mm

Quality Distribution:
"""
        
        # Quality assessment
        excellent = (df['total_deviation_mm'] <= 25).sum()
        good = ((df['total_deviation_mm'] > 25) & (df['total_deviation_mm'] <= 50)).sum()
        acceptable = ((df['total_deviation_mm'] > 50) & (df['total_deviation_mm'] <= 75)).sum()
        poor = (df['total_deviation_mm'] > 75).sum()
        total = len(df)
        
        stats_text += f"""
Excellent (≤25mm): {excellent} ({excellent/total:.1%})
Good (25-50mm): {good} ({good/total:.1%})
Acceptable (50-75mm): {acceptable} ({acceptable/total:.1%})
Poor (>75mm): {poor} ({poor/total:.1%})
"""
        
        axes[1,1].text(0.1, 0.9, stats_text, transform=axes[1,1].transAxes, 
                      fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('real_mtbm_deviation_analysis.png', dpi=300, bbox_inches='tight')
            print("Saved: real_mtbm_deviation_analysis.png")
        
        plt.show()
    
    def generate_data_summary(self, df):
        """
        Generate comprehensive summary of the real data
        """
        print("\n" + "="*80)
        print("REAL MTBM DATA ANALYSIS SUMMARY")
        print("="*80)
        
        print(f"\n1. DATA OVERVIEW")
        print(f"   Total Records: {len(df):,}")
        if 'timestamp' in df.columns:
            print(f"   Time Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
        if 'tunnel_length_m' in df.columns:
            print(f"   Tunnel Length: {df['tunnel_length_m'].max():.1f} meters")
        
        print(f"\n2. AVAILABLE PARAMETERS")
        for i, col in enumerate(df.columns, 1):
            non_null_count = df[col].notna().sum()
            percentage = non_null_count / len(df) * 100
            print(f"   {i:2d}. {col}: {non_null_count:,} records ({percentage:.1f}%)")
        
        # Performance metrics (if available)
        if 'advance_speed_mm_min' in df.columns:
            print(f"\n3. PERFORMANCE METRICS")
            print(f"   Average Advance Speed: {df['advance_speed_mm_min'].mean():.1f} mm/min")
            print(f"   Max Advance Speed: {df['advance_speed_mm_min'].max():.1f} mm/min")
            print(f"   Min Advance Speed: {df['advance_speed_mm_min'].min():.1f} mm/min")
        
        if 'working_pressure_bar' in df.columns:
            print(f"   Average Working Pressure: {df['working_pressure_bar'].mean():.1f} bar")
        
        if 'drilling_efficiency' in df.columns:
            print(f"   Average Drilling Efficiency: {df['drilling_efficiency'].mean():.3f} mm/min/bar")
        
        # Deviation analysis (if available)
        if 'total_deviation_mm' in df.columns:
            print(f"\n4. DEVIATION ANALYSIS")
            print(f"   Average Total Deviation: {df['total_deviation_mm'].mean():.1f} mm")
            print(f"   Maximum Total Deviation: {df['total_deviation_mm'].max():.1f} mm")
            print(f"   Standard Deviation: {df['total_deviation_mm'].std():.1f} mm")
        
        print("="*80)

def main():
    """
    Main execution function
    """
    print("Real MTBM Data Plotting Framework")
    print("=================================")
    
    # Check for command line argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
    else:
        # Look for common MTBM data files
        possible_files = [
            'mtbm_data.csv',
            'tunnel_data.csv',
            'AVN3000_Data.xlsx',
            'protocol_data.xlsx',
            'mtbm_comprehensive_data.csv'
        ]
        
        file_path = None
        for filename in possible_files:
            if os.path.exists(filename):
                file_path = filename
                break
        
        if not file_path:
            print("No MTBM data file found!")
            print("Usage: python plot_real_mtbm_data.py [data_file.csv]")
            print("\nOr place one of these files in the current directory:")
            for filename in possible_files:
                print(f"  - {filename}")
            return
    
    # Initialize plotter
    plotter = RealMTBMDataPlotter()
    
    # Load and prepare data
    df = plotter.load_and_prepare_data(file_path)
    
    if df is None:
        print("Failed to load data file")
        return
    
    # Generate visualizations
    plotter.plot_key_parameters(df)
    plotter.plot_deviation_analysis(df)
    
    # Generate summary
    plotter.generate_data_summary(df)
    
    print("\nReal data analysis completed!")

if __name__ == "__main__":
    main()
