#!/usr/bin/env python3
"""
MTBM Deviation Visualization Tool
Generate comprehensive graphs for deviation analysis from MTBM protocol data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class MTBMDeviationVisualizer:
    """
    Comprehensive visualization tool for MTBM deviation analysis
    """
    
    def __init__(self, data_path=None):
        """Initialize with optional data path"""
        self.data_path = data_path
        self.df = None
        self.processed_df = None
        
    def load_data(self, csv_path=None):
        """Load MTBM protocol data"""
        
        if csv_path is None:
            csv_path = self.data_path
            
        if csv_path is None:
            # Create sample data for demonstration
            self.df = self.create_sample_deviation_data()
            print("Using sample deviation data for demonstration")
        else:
            # Load real data
            columns = [
                'date', 'time', 'tunnel_length', 'hor_dev_machine', 'vert_dev_machine',
                'hor_dev_drill_head', 'vert_dev_drill_head', 'yaw', 'pitch', 'roll',
                'temperature', 'survey_mode', 'sc_cyl_01', 'sc_cyl_02', 'sc_cyl_03',
                'sc_cyl_04', 'advance_speed', 'interjack_force', 'interjack_active',
                'working_pressure', 'revolution_rpm', 'earth_pressure', 'total_force'
            ]
            
            self.df = pd.read_csv(csv_path, names=columns, skiprows=1)
            self.df['datetime'] = pd.to_datetime(self.df['date'] + ' ' + self.df['time'], errors='coerce')
            self.df = self.df.dropna(subset=['datetime']).sort_values('datetime').reset_index(drop=True)
            print(f"Loaded {len(self.df)} readings from {csv_path}")
        
        # Process the data
        self.process_deviation_features()
        
    def create_sample_deviation_data(self):
        """Create realistic sample deviation data for demonstration"""
        
        np.random.seed(42)
        n_samples = 200
        
        # Simulate realistic tunneling scenario with various deviation patterns
        tunnel_length = np.cumsum(np.random.normal(0.08, 0.02, n_samples))  # Progressive tunnel length
        
        # Create different deviation scenarios
        scenarios = []
        
        # Scenario 1: Good alignment (first 50 readings)
        good_h = np.random.normal(0, 3, 50)
        good_v = np.random.normal(0, 2, 50)
        scenarios.extend([(h, v) for h, v in zip(good_h, good_v)])
        
        # Scenario 2: Gradual drift (next 50 readings)
        drift_h = np.random.normal(0, 3, 50) + np.linspace(0, 15, 50)
        drift_v = np.random.normal(0, 2, 50) + np.linspace(0, -10, 50)
        scenarios.extend([(h, v) for h, v in zip(drift_h, drift_v)])
        
        # Scenario 3: Correction period (next 50 readings)
        correct_h = np.linspace(15, 2, 50) + np.random.normal(0, 2, 50)
        correct_v = np.linspace(-10, 1, 50) + np.random.normal(0, 1.5, 50)
        scenarios.extend([(h, v) for h, v in zip(correct_h, correct_v)])
        
        # Scenario 4: Challenging conditions (final 50 readings)
        challenge_h = np.random.normal(0, 8, 50) + np.sin(np.linspace(0, 4*np.pi, 50)) * 5
        challenge_v = np.random.normal(0, 6, 50) + np.cos(np.linspace(0, 3*np.pi, 50)) * 4
        scenarios.extend([(h, v) for h, v in zip(challenge_h, challenge_v)])
        
        # Create DataFrame
        data = {
            'reading_id': range(1, n_samples + 1),
            'tunnel_length': tunnel_length,
            'hor_dev_machine': [s[0] for s in scenarios],
            'vert_dev_machine': [s[1] for s in scenarios],
            'hor_dev_drill_head': [s[0] + np.random.normal(0, 2) for s in scenarios],
            'vert_dev_drill_head': [s[1] + np.random.normal(0, 2) for s in scenarios],
        }
        
        # Add timestamps
        start_time = datetime(2023, 7, 1, 8, 0, 0)
        data['datetime'] = [start_time + timedelta(minutes=i*5) for i in range(n_samples)]
        
        return pd.DataFrame(data)
    
    def process_deviation_features(self):
        """Process deviation data and create engineered features"""
        
        df = self.df.copy()
        
        # Core deviation calculations
        df['total_deviation_machine'] = np.sqrt(df['hor_dev_machine']**2 + df['vert_dev_machine']**2)
        df['total_deviation_drill_head'] = np.sqrt(df['hor_dev_drill_head']**2 + df['vert_dev_drill_head']**2)
        df['deviation_difference'] = df['total_deviation_drill_head'] - df['total_deviation_machine']
        df['alignment_quality'] = 1 / (1 + df['total_deviation_machine'])
        
        # Deviation trends
        window = 10
        df['deviation_trend'] = df['total_deviation_machine'].rolling(window=window, min_periods=1).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0
        )
        
        # Moving averages
        df['total_dev_ma5'] = df['total_deviation_machine'].rolling(window=5, min_periods=1).mean()
        df['total_dev_ma10'] = df['total_deviation_machine'].rolling(window=10, min_periods=1).mean()
        
        # Deviation categories for analysis
        df['deviation_category'] = pd.cut(
            df['total_deviation_machine'], 
            bins=[-np.inf, 5, 15, 25, np.inf],
            labels=['Excellent (<5mm)', 'Good (5-15mm)', 'Acceptable (15-25mm)', 'Poor (>25mm)']
        )
        
        # Quality grades
        df['quality_grade'] = pd.cut(
            df['alignment_quality'],
            bins=[0, 0.3, 0.5, 0.8, 1.0],
            labels=['Poor', 'Acceptable', 'Good', 'Excellent']
        )
        
        self.processed_df = df
        print(f"Processed {len(df)} readings with deviation features")
    
    def plot_deviation_over_distance(self, save_path=None):
        """Plot deviation trends over tunnel distance"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        df = self.processed_df
        
        # Plot 1: Raw horizontal and vertical deviations
        axes[0,0].plot(df['tunnel_length'], df['hor_dev_machine'], 
                      label='Horizontal (Machine)', alpha=0.7, linewidth=1.5)
        axes[0,0].plot(df['tunnel_length'], df['vert_dev_machine'], 
                      label='Vertical (Machine)', alpha=0.7, linewidth=1.5)
        axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0,0].axhline(y=10, color='orange', linestyle='--', alpha=0.5, label='±10mm threshold')
        axes[0,0].axhline(y=-10, color='orange', linestyle='--', alpha=0.5)
        axes[0,0].set_xlabel('Tunnel Length (m)')
        axes[0,0].set_ylabel('Deviation (mm)')
        axes[0,0].set_title('Raw Horizontal & Vertical Deviations')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Plot 2: Total deviation with moving averages
        axes[0,1].plot(df['tunnel_length'], df['total_deviation_machine'], 
                      label='Total Deviation', alpha=0.6, linewidth=1)
        axes[0,1].plot(df['tunnel_length'], df['total_dev_ma5'], 
                      label='5-Point Moving Average', linewidth=2)
        axes[0,1].plot(df['tunnel_length'], df['total_dev_ma10'], 
                      label='10-Point Moving Average', linewidth=2)
        
        # Add threshold lines
        axes[0,1].axhline(y=5, color='green', linestyle='--', alpha=0.7, label='Excellent (<5mm)')
        axes[0,1].axhline(y=15, color='orange', linestyle='--', alpha=0.7, label='Good (<15mm)')
        axes[0,1].axhline(y=25, color='red', linestyle='--', alpha=0.7, label='Acceptable (<25mm)')
        
        axes[0,1].set_xlabel('Tunnel Length (m)')
        axes[0,1].set_ylabel('Total Deviation (mm)')
        axes[0,1].set_title('Total Deviation with Moving Averages')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Machine vs Drill Head comparison
        axes[1,0].plot(df['tunnel_length'], df['total_deviation_machine'], 
                      label='Machine Body', linewidth=2)
        axes[1,0].plot(df['tunnel_length'], df['total_deviation_drill_head'], 
                      label='Drill Head', linewidth=2, alpha=0.8)
        axes[1,0].fill_between(df['tunnel_length'], 
                              df['total_deviation_machine'], 
                              df['total_deviation_drill_head'],
                              alpha=0.2, label='Alignment Difference')
        axes[1,0].set_xlabel('Tunnel Length (m)')
        axes[1,0].set_ylabel('Total Deviation (mm)')
        axes[1,0].set_title('Machine Body vs Drill Head Alignment')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 4: Alignment quality score
        axes[1,1].plot(df['tunnel_length'], df['alignment_quality'], 
                      color='purple', linewidth=2)
        axes[1,1].fill_between(df['tunnel_length'], 0, df['alignment_quality'], 
                              alpha=0.3, color='purple')
        
        # Add quality thresholds
        axes[1,1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7, label='Excellent (>0.8)')
        axes[1,1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7, label='Good (>0.5)')
        axes[1,1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7, label='Acceptable (>0.3)')
        
        axes[1,1].set_xlabel('Tunnel Length (m)')
        axes[1,1].set_ylabel('Alignment Quality Score')
        axes[1,1].set_title('Alignment Quality Over Distance')
        axes[1,1].set_ylim(0, 1)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved deviation over distance plot to: {save_path}")
        else:
            plt.show()
    
    def plot_deviation_patterns(self, save_path=None):
        """Plot deviation pattern analysis"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        df = self.processed_df
        
        # Plot 1: Deviation scatter plot (trajectory view)
        scatter = axes[0,0].scatter(df['hor_dev_machine'], df['vert_dev_machine'], 
                                   c=df['tunnel_length'], cmap='viridis', alpha=0.6, s=30)
        axes[0,0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0,0].axvline(x=0, color='black', linestyle='-', alpha=0.3)
        
        # Add circles for deviation thresholds
        circle1 = plt.Circle((0, 0), 5, fill=False, color='green', linestyle='--', alpha=0.7)
        circle2 = plt.Circle((0, 0), 15, fill=False, color='orange', linestyle='--', alpha=0.7)
        circle3 = plt.Circle((0, 0), 25, fill=False, color='red', linestyle='--', alpha=0.7)
        axes[0,0].add_patch(circle1)
        axes[0,0].add_patch(circle2)
        axes[0,0].add_patch(circle3)
        
        axes[0,0].set_xlabel('Horizontal Deviation (mm)')
        axes[0,0].set_ylabel('Vertical Deviation (mm)')
        axes[0,0].set_title('Deviation Trajectory (Color = Distance)')
        axes[0,0].grid(True, alpha=0.3)
        axes[0,0].set_aspect('equal')
        plt.colorbar(scatter, ax=axes[0,0], label='Tunnel Length (m)')
        
        # Plot 2: Deviation distribution histogram
        axes[0,1].hist(df['total_deviation_machine'], bins=30, alpha=0.7, 
                      color='skyblue', edgecolor='black')
        axes[0,1].axvline(df['total_deviation_machine'].mean(), 
                         color='red', linestyle='--', linewidth=2, 
                         label=f'Mean: {df["total_deviation_machine"].mean():.1f}mm')
        axes[0,1].axvline(df['total_deviation_machine'].median(), 
                         color='green', linestyle='--', linewidth=2,
                         label=f'Median: {df["total_deviation_machine"].median():.1f}mm')
        axes[0,1].set_xlabel('Total Deviation (mm)')
        axes[0,1].set_ylabel('Frequency')
        axes[0,1].set_title('Deviation Distribution')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Plot 3: Box plot by deviation category
        category_data = [df[df['deviation_category'] == cat]['total_deviation_machine'].values 
                        for cat in df['deviation_category'].cat.categories]
        box_plot = axes[0,2].boxplot(category_data, labels=df['deviation_category'].cat.categories,
                                    patch_artist=True)
        
        # Color the boxes
        colors = ['green', 'orange', 'red', 'darkred']
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        axes[0,2].set_ylabel('Total Deviation (mm)')
        axes[0,2].set_title('Deviation by Quality Category')
        axes[0,2].tick_params(axis='x', rotation=45)
        axes[0,2].grid(True, alpha=0.3)
        
        # Plot 4: Deviation trend analysis
        axes[1,0].plot(df['tunnel_length'], df['deviation_trend'], 
                      color='purple', linewidth=2)
        axes[1,0].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,0].fill_between(df['tunnel_length'], 0, df['deviation_trend'], 
                              where=(df['deviation_trend'] > 0), 
                              color='red', alpha=0.3, label='Worsening')
        axes[1,0].fill_between(df['tunnel_length'], 0, df['deviation_trend'], 
                              where=(df['deviation_trend'] <= 0), 
                              color='green', alpha=0.3, label='Improving')
        axes[1,0].set_xlabel('Tunnel Length (m)')
        axes[1,0].set_ylabel('Deviation Trend (mm/reading)')
        axes[1,0].set_title('Deviation Trend Analysis')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Plot 5: Alignment difference analysis
        axes[1,1].plot(df['tunnel_length'], df['deviation_difference'], 
                      color='brown', linewidth=1.5)
        axes[1,1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1,1].fill_between(df['tunnel_length'], 0, df['deviation_difference'], 
                              where=(df['deviation_difference'] > 0), 
                              color='orange', alpha=0.3, label='Drill Head Worse')
        axes[1,1].fill_between(df['tunnel_length'], 0, df['deviation_difference'], 
                              where=(df['deviation_difference'] <= 0), 
                              color='blue', alpha=0.3, label='Machine Body Worse')
        axes[1,1].set_xlabel('Tunnel Length (m)')
        axes[1,1].set_ylabel('Deviation Difference (mm)')
        axes[1,1].set_title('Machine vs Drill Head Alignment')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        # Plot 6: Quality grade pie chart
        quality_counts = df['quality_grade'].value_counts()
        colors_pie = ['darkred', 'red', 'orange', 'green']
        axes[1,2].pie(quality_counts.values, labels=quality_counts.index, 
                     colors=colors_pie, autopct='%1.1f%%', startangle=90)
        axes[1,2].set_title('Overall Quality Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved deviation patterns plot to: {save_path}")
        else:
            plt.show()
    
    def plot_time_series_analysis(self, save_path=None):
        """Plot time-based deviation analysis"""
        
        if 'datetime' not in self.processed_df.columns:
            print("No datetime information available for time series analysis")
            return
            
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        df = self.processed_df
        
        # Plot 1: Time series of all deviations
        axes[0].plot(df['datetime'], df['hor_dev_machine'], 
                    label='Horizontal', alpha=0.7, linewidth=1)
        axes[0].plot(df['datetime'], df['vert_dev_machine'], 
                    label='Vertical', alpha=0.7, linewidth=1)
        axes[0].plot(df['datetime'], df['total_deviation_machine'], 
                    label='Total', linewidth=2, color='red')
        
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].set_ylabel('Deviation (mm)')
        axes[0].set_title('Deviation Time Series')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Quality score over time
        axes[1].plot(df['datetime'], df['alignment_quality'], 
                    color='purple', linewidth=2)
        axes[1].fill_between(df['datetime'], 0, df['alignment_quality'], 
                           alpha=0.3, color='purple')
        
        # Add quality thresholds
        axes[1].axhline(y=0.8, color='green', linestyle='--', alpha=0.7)
        axes[1].axhline(y=0.5, color='orange', linestyle='--', alpha=0.7)
        axes[1].axhline(y=0.3, color='red', linestyle='--', alpha=0.7)
        
        axes[1].set_ylabel('Alignment Quality')
        axes[1].set_title('Alignment Quality Over Time')
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Deviation trend over time
        axes[2].plot(df['datetime'], df['deviation_trend'], 
                    color='brown', linewidth=2)
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[2].fill_between(df['datetime'], 0, df['deviation_trend'], 
                           where=(df['deviation_trend'] > 0), 
                           color='red', alpha=0.3, label='Deteriorating')
        axes[2].fill_between(df['datetime'], 0, df['deviation_trend'], 
                           where=(df['deviation_trend'] <= 0), 
                           color='green', alpha=0.3, label='Improving')
        
        axes[2].set_xlabel('Time')
        axes[2].set_ylabel('Deviation Trend')
        axes[2].set_title('Deviation Trend Over Time')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved time series analysis plot to: {save_path}")
        else:
            plt.show()
    
    def generate_deviation_report(self):
        """Generate comprehensive deviation analysis report"""
        
        df = self.processed_df
        
        print("="*80)
        print("MTBM DEVIATION ANALYSIS REPORT")
        print("="*80)
        
        # Basic statistics
        print("\n1. BASIC DEVIATION STATISTICS")
        print("-" * 50)
        print(f"Total readings: {len(df)}")
        print(f"Tunnel length: {df['tunnel_length'].max():.2f}m")
        
        print(f"\nHorizontal Deviation (mm):")
        print(f"  Mean: {df['hor_dev_machine'].mean():6.2f}")
        print(f"  Std:  {df['hor_dev_machine'].std():6.2f}")
        print(f"  Range: {df['hor_dev_machine'].min():6.2f} to {df['hor_dev_machine'].max():6.2f}")
        
        print(f"\nVertical Deviation (mm):")
        print(f"  Mean: {df['vert_dev_machine'].mean():6.2f}")
        print(f"  Std:  {df['vert_dev_machine'].std():6.2f}")
        print(f"  Range: {df['vert_dev_machine'].min():6.2f} to {df['vert_dev_machine'].max():6.2f}")
        
        print(f"\nTotal Deviation (mm):")
        print(f"  Mean: {df['total_deviation_machine'].mean():6.2f}")
        print(f"  Std:  {df['total_deviation_machine'].std():6.2f}")
        print(f"  Range: {df['total_deviation_machine'].min():6.2f} to {df['total_deviation_machine'].max():6.2f}")
        
        # Quality analysis
        print("\n2. ALIGNMENT QUALITY ANALYSIS")
        print("-" * 50)
        quality_counts = df['quality_grade'].value_counts()
        total_readings = len(df)
        
        for grade in ['Excellent', 'Good', 'Acceptable', 'Poor']:
            if grade in quality_counts.index:
                count = quality_counts[grade]
                percentage = (count / total_readings) * 100
                print(f"{grade:12}: {count:4d} readings ({percentage:5.1f}%)")
        
        print(f"\nAverage Quality Score: {df['alignment_quality'].mean():.3f}")
        
        # Trend analysis
        print("\n3. DEVIATION TREND ANALYSIS")
        print("-" * 50)
        
        improving_readings = (df['deviation_trend'] < -0.1).sum()
        stable_readings = ((df['deviation_trend'] >= -0.1) & (df['deviation_trend'] <= 0.1)).sum()
        worsening_readings = (df['deviation_trend'] > 0.1).sum()
        
        print(f"Improving trend:  {improving_readings:4d} readings ({improving_readings/total_readings*100:5.1f}%)")
        print(f"Stable trend:     {stable_readings:4d} readings ({stable_readings/total_readings*100:5.1f}%)")
        print(f"Worsening trend:  {worsening_readings:4d} readings ({worsening_readings/total_readings*100:5.1f}%)")
        
        # Alignment analysis
        print("\n4. MACHINE VS DRILL HEAD ALIGNMENT")
        print("-" * 50)
        
        avg_diff = df['deviation_difference'].mean()
        print(f"Average alignment difference: {avg_diff:6.2f}mm")
        
        if avg_diff > 2:
            print("  → Drill head consistently worse than machine body")
        elif avg_diff < -2:
            print("  → Machine body consistently worse than drill head")
        else:
            print("  → Good overall alignment between components")
        
        well_aligned = (abs(df['deviation_difference']) < 2).sum()
        print(f"Well-aligned readings: {well_aligned} ({well_aligned/total_readings*100:.1f}%)")
        
        # Recommendations
        print("\n5. RECOMMENDATIONS")
        print("-" * 50)
        
        mean_deviation = df['total_deviation_machine'].mean()
        if mean_deviation > 20:
            print("• HIGH PRIORITY: Average deviation exceeds 20mm - immediate steering system review required")
        elif mean_deviation > 15:
            print("• MEDIUM PRIORITY: Average deviation exceeds 15mm - enhanced monitoring recommended")
        else:
            print("• Overall deviation levels within acceptable range")
        
        if df['alignment_quality'].mean() < 0.3:
            print("• Poor alignment quality - consider operator training and equipment calibration")
        elif df['alignment_quality'].mean() < 0.5:
            print("• Moderate alignment quality - regular monitoring and adjustments needed")
        
        if worsening_readings > improving_readings:
            print("• Trend analysis shows more deteriorating than improving periods - investigate causes")
        
        print("\n" + "="*80)
    
    def generate_all_deviation_plots(self, output_dir="deviation_plots"):
        """Generate all deviation visualization plots"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"Generating comprehensive deviation plots in {output_dir}/...")
        
        # Generate all plots
        self.plot_deviation_over_distance(f"{output_dir}/deviation_over_distance.png")
        self.plot_deviation_patterns(f"{output_dir}/deviation_patterns.png")
        self.plot_time_series_analysis(f"{output_dir}/time_series_analysis.png")
        
        # Generate report
        self.generate_deviation_report()
        
        print(f"\nAll plots saved to {output_dir}/ directory")
        print("Generated files:")
        print("- deviation_over_distance.png")
        print("- deviation_patterns.png") 
        print("- time_series_analysis.png")


def main():
    """Main function to demonstrate deviation visualization"""
    
    print("MTBM DEVIATION VISUALIZATION TOOL")
    print("="*50)
    
    # Initialize visualizer
    visualizer = MTBMDeviationVisualizer()
    
    # Try to load real data first, fall back to sample data
    try:
        # Try to load AVN1200 data
        visualizer.load_data("AVN1200-ML/measure_protocol_original_.xls.csv")
    except:
        try:
            # Try alternative path
            visualizer.load_data("measure_protocol_original_.xls.csv") 
        except:
            # Use sample data
            print("Real data not found, using sample data for demonstration")
            visualizer.load_data()
    
    # Generate all visualizations
    visualizer.generate_all_deviation_plots()
    
    return visualizer


if __name__ == "__main__":
    viz = main()