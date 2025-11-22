#!/usr/bin/env python3
"""
Simple MTBM Deviation Graph Generator
Creates deviation analysis graphs using basic Python only
"""

import math
import csv
from datetime import datetime, timedelta

class SimpleDeviationGraphGenerator:
    """
    Generate deviation graphs and analysis using basic Python
    """
    
    def __init__(self):
        self.data = []
        self.processed_data = []
    
    def load_sample_data(self):
        """Create sample deviation data for demonstration"""
        print("Generating sample MTBM deviation data...")
        
        # Create 100 sample readings with realistic deviation patterns
        sample_data = []
        
        for i in range(100):
            tunnel_length = i * 0.1  # 10cm per reading
            
            # Create different scenarios
            if i < 25:  # Good alignment period
                h_dev = (i % 5 - 2) + (i % 3 - 1) * 0.5
                v_dev = (i % 4 - 1.5) + (i % 2 - 0.5) * 0.3
            elif i < 50:  # Gradual drift
                h_dev = (i - 25) * 0.3 + (i % 5 - 2)
                v_dev = -(i - 25) * 0.2 + (i % 4 - 1.5)
            elif i < 75:  # Correction period
                h_dev = (75 - i) * 0.2 + (i % 3 - 1)
                v_dev = (75 - i) * 0.15 + (i % 4 - 1.5)
            else:  # Challenging conditions
                h_dev = 5 * math.sin((i - 75) * 0.3) + (i % 5 - 2)
                v_dev = 3 * math.cos((i - 75) * 0.4) + (i % 3 - 1)
            
            # Add some noise
            h_dev += (i * 17 % 7 - 3) * 0.2
            v_dev += (i * 23 % 5 - 2) * 0.15
            
            # Drill head typically has slightly more deviation
            h_drill = h_dev + (i % 7 - 3) * 0.3
            v_drill = v_dev + (i % 6 - 2.5) * 0.25
            
            sample_data.append({
                'reading_id': i + 1,
                'tunnel_length': tunnel_length,
                'hor_dev_machine': h_dev,
                'vert_dev_machine': v_dev,
                'hor_dev_drill_head': h_drill,
                'vert_dev_drill_head': v_drill
            })
        
        self.data = sample_data
        print(f"Generated {len(sample_data)} sample readings")
        
    def load_real_data(self, csv_path):
        """Load real MTBM data from CSV file"""
        try:
            with open(csv_path, 'r') as file:
                reader = csv.reader(file)
                next(reader)  # Skip header
                
                data = []
                for i, row in enumerate(reader):
                    if len(row) >= 7:  # Ensure we have enough columns
                        try:
                            data.append({
                                'reading_id': i + 1,
                                'tunnel_length': float(row[2]),
                                'hor_dev_machine': float(row[3]),
                                'vert_dev_machine': float(row[4]),
                                'hor_dev_drill_head': float(row[5]),
                                'vert_dev_drill_head': float(row[6])
                            })
                        except ValueError:
                            continue  # Skip invalid rows
                
                self.data = data
                print(f"Loaded {len(data)} readings from {csv_path}")
                return True
                
        except FileNotFoundError:
            print(f"File {csv_path} not found")
            return False
        except Exception as e:
            print(f"Error loading {csv_path}: {e}")
            return False
    
    def process_deviation_features(self):
        """Calculate deviation features"""
        print("Processing deviation features...")
        
        processed = []
        
        for reading in self.data:
            # Calculate total deviations
            total_machine = math.sqrt(reading['hor_dev_machine']**2 + reading['vert_dev_machine']**2)
            total_drill = math.sqrt(reading['hor_dev_drill_head']**2 + reading['vert_dev_drill_head']**2)
            
            # Calculate deviation difference
            deviation_diff = total_drill - total_machine
            
            # Calculate alignment quality
            alignment_quality = 1 / (1 + total_machine)
            
            # Determine quality grade
            if alignment_quality > 0.8:
                quality_grade = "Excellent"
            elif alignment_quality > 0.5:
                quality_grade = "Good"
            elif alignment_quality > 0.3:
                quality_grade = "Acceptable"
            else:
                quality_grade = "Poor"
            
            # Determine deviation category
            if total_machine < 5:
                dev_category = "Excellent (<5mm)"
            elif total_machine < 15:
                dev_category = "Good (5-15mm)"
            elif total_machine < 25:
                dev_category = "Acceptable (15-25mm)"
            else:
                dev_category = "Poor (>25mm)"
            
            processed_reading = reading.copy()
            processed_reading.update({
                'total_deviation_machine': total_machine,
                'total_deviation_drill_head': total_drill,
                'deviation_difference': deviation_diff,
                'alignment_quality': alignment_quality,
                'quality_grade': quality_grade,
                'deviation_category': dev_category
            })
            
            processed.append(processed_reading)
        
        self.processed_data = processed
        print(f"Processed {len(processed)} readings with deviation features")
    
    def create_ascii_plots(self):
        """Create ASCII-based plots for deviation visualization"""
        
        print("\n" + "="*80)
        print("MTBM DEVIATION ANALYSIS - ASCII PLOTS")
        print("="*80)
        
        data = self.processed_data
        
        # Plot 1: Total Deviation Over Distance
        print("\n1. TOTAL DEVIATION OVER TUNNEL DISTANCE")
        print("-" * 60)
        
        # Find min/max for scaling
        max_dev = max(reading['total_deviation_machine'] for reading in data)
        max_length = max(reading['tunnel_length'] for reading in data)
        
        print(f"Distance: 0.0m to {max_length:.1f}m | Deviation: 0.0mm to {max_dev:.1f}mm")
        print()
        
        # Create ASCII plot (simplified)
        plot_width = 60
        plot_height = 15
        
        # Create grid
        grid = [[' ' for _ in range(plot_width)] for _ in range(plot_height)]
        
        # Plot data points
        for reading in data:
            x = int((reading['tunnel_length'] / max_length) * (plot_width - 1))
            y = plot_height - 1 - int((reading['total_deviation_machine'] / max_dev) * (plot_height - 1))
            
            if 0 <= x < plot_width and 0 <= y < plot_height:
                if reading['quality_grade'] == 'Excellent':
                    grid[y][x] = '●'
                elif reading['quality_grade'] == 'Good':
                    grid[y][x] = '○'
                elif reading['quality_grade'] == 'Acceptable':
                    grid[y][x] = '△'
                else:
                    grid[y][x] = '×'
        
        # Print grid
        for i, row in enumerate(grid):
            dev_val = (plot_height - 1 - i) * max_dev / (plot_height - 1)
            print(f"{dev_val:5.1f}|{''.join(row)}")
        
        print("     " + "-" * plot_width)
        print(f"      0.0m{' ' * (plot_width - 15)}{max_length:.1f}m")
        print("\nLegend: ● Excellent  ○ Good  △ Acceptable  × Poor")
        
        # Plot 2: Deviation Distribution Summary
        print("\n2. DEVIATION QUALITY DISTRIBUTION")
        print("-" * 60)
        
        # Count by quality grade
        quality_counts = {'Excellent': 0, 'Good': 0, 'Acceptable': 0, 'Poor': 0}
        for reading in data:
            quality_counts[reading['quality_grade']] += 1
        
        total_readings = len(data)
        
        print("Quality Grade Breakdown:")
        for grade, count in quality_counts.items():
            percentage = (count / total_readings) * 100
            bar_length = int(percentage / 2)  # Scale to fit
            bar = '█' * bar_length
            print(f"{grade:10}: {count:3d} ({percentage:5.1f}%) {bar}")
        
        # Plot 3: Horizontal vs Vertical Deviation Pattern
        print("\n3. HORIZONTAL VS VERTICAL DEVIATION PATTERN")
        print("-" * 60)
        
        # Find min/max for both axes
        h_values = [reading['hor_dev_machine'] for reading in data]
        v_values = [reading['vert_dev_machine'] for reading in data]
        
        h_min, h_max = min(h_values), max(h_values)
        v_min, v_max = min(v_values), max(v_values)
        
        print(f"Horizontal: {h_min:.1f}mm to {h_max:.1f}mm | Vertical: {v_min:.1f}mm to {v_max:.1f}mm")
        print()
        
        # Create scatter plot
        plot_size = 20
        scatter_grid = [[' ' for _ in range(plot_size)] for _ in range(plot_size)]
        
        for reading in data:
            if h_max != h_min and v_max != v_min:
                x = int(((reading['hor_dev_machine'] - h_min) / (h_max - h_min)) * (plot_size - 1))
                y = plot_size - 1 - int(((reading['vert_dev_machine'] - v_min) / (v_max - v_min)) * (plot_size - 1))
                
                if 0 <= x < plot_size and 0 <= y < plot_size:
                    if scatter_grid[y][x] == ' ':
                        scatter_grid[y][x] = '●'
                    else:
                        scatter_grid[y][x] = '█'  # Multiple points
        
        # Print scatter plot
        for i, row in enumerate(scatter_grid):
            v_val = v_min + (plot_size - 1 - i) * (v_max - v_min) / (plot_size - 1)
            print(f"{v_val:5.1f}|{''.join(row)}|")
        
        print("     " + "-" * (plot_size + 2))
        print(f"     {h_min:5.1f}{' ' * (plot_size - 10)}{h_max:5.1f}")
        print("          Horizontal Deviation (mm)")
        
    def generate_statistics_report(self):
        """Generate detailed statistics report"""
        
        print("\n" + "="*80)
        print("DEVIATION STATISTICS REPORT")
        print("="*80)
        
        data = self.processed_data
        
        # Basic statistics
        h_values = [reading['hor_dev_machine'] for reading in data]
        v_values = [reading['vert_dev_machine'] for reading in data]
        total_values = [reading['total_deviation_machine'] for reading in data]
        quality_values = [reading['alignment_quality'] for reading in data]
        
        def calculate_stats(values):
            n = len(values)
            mean_val = sum(values) / n
            variance = sum((x - mean_val)**2 for x in values) / n
            std_dev = math.sqrt(variance)
            sorted_vals = sorted(values)
            median = sorted_vals[n//2] if n % 2 == 1 else (sorted_vals[n//2-1] + sorted_vals[n//2]) / 2
            return {
                'mean': mean_val,
                'std': std_dev,
                'min': min(values),
                'max': max(values),
                'median': median
            }
        
        print(f"\nTotal Readings: {len(data)}")
        print(f"Tunnel Length: {data[-1]['tunnel_length']:.2f}m")
        
        print(f"\nHORIZONTAL DEVIATION STATISTICS:")
        h_stats = calculate_stats(h_values)
        for key, value in h_stats.items():
            print(f"  {key.capitalize():8}: {value:8.2f}mm")
        
        print(f"\nVERTICAL DEVIATION STATISTICS:")
        v_stats = calculate_stats(v_values)
        for key, value in v_stats.items():
            print(f"  {key.capitalize():8}: {value:8.2f}mm")
        
        print(f"\nTOTAL DEVIATION STATISTICS:")
        t_stats = calculate_stats(total_values)
        for key, value in t_stats.items():
            print(f"  {key.capitalize():8}: {value:8.2f}mm")
        
        print(f"\nALIGNMENT QUALITY STATISTICS:")
        q_stats = calculate_stats(quality_values)
        for key, value in q_stats.items():
            print(f"  {key.capitalize():8}: {value:8.3f}")
        
        # Quality distribution
        print(f"\nQUALITY GRADE DISTRIBUTION:")
        quality_counts = {'Excellent': 0, 'Good': 0, 'Acceptable': 0, 'Poor': 0}
        for reading in data:
            quality_counts[reading['quality_grade']] += 1
        
        total_readings = len(data)
        for grade, count in quality_counts.items():
            percentage = (count / total_readings) * 100
            print(f"  {grade:10}: {count:4d} readings ({percentage:5.1f}%)")
        
        # Machine vs Drill Head Analysis
        diff_values = [reading['deviation_difference'] for reading in data]
        diff_stats = calculate_stats(diff_values)
        
        print(f"\nMACHINE vs DRILL HEAD ALIGNMENT:")
        print(f"  Average difference: {diff_stats['mean']:6.2f}mm")
        if diff_stats['mean'] > 2:
            print(f"  → Drill head consistently worse alignment")
        elif diff_stats['mean'] < -2:
            print(f"  → Machine body consistently worse alignment")
        else:
            print(f"  → Good overall component alignment")
        
        # Recommendations
        print(f"\nRECOMMENDALTIONS:")
        mean_total_dev = t_stats['mean']
        mean_quality = q_stats['mean']
        
        if mean_total_dev > 20:
            print("  • HIGH PRIORITY: Average deviation >20mm - immediate steering review required")
        elif mean_total_dev > 15:
            print("  • MEDIUM PRIORITY: Average deviation >15mm - enhanced monitoring needed")
        else:
            print("  • Deviation levels within acceptable operational range")
        
        if mean_quality < 0.3:
            print("  • Poor alignment quality - consider equipment calibration")
        elif mean_quality < 0.5:
            print("  • Moderate alignment quality - regular adjustments recommended")
        else:
            print("  • Good alignment quality maintained")
        
        poor_percentage = (quality_counts['Poor'] / total_readings) * 100
        if poor_percentage > 25:
            print("  • High percentage of poor readings - investigate systematic issues")
    
    def export_data_summary(self, filename="deviation_analysis.csv"):
        """Export processed data to CSV for further analysis"""
        
        try:
            with open(filename, 'w', newline='') as file:
                writer = csv.writer(file)
                
                # Write header
                header = [
                    'Reading_ID', 'Tunnel_Length', 'Horizontal_Dev', 'Vertical_Dev',
                    'Total_Deviation', 'Alignment_Quality', 'Quality_Grade', 
                    'Deviation_Category', 'Deviation_Difference'
                ]
                writer.writerow(header)
                
                # Write data
                for reading in self.processed_data:
                    row = [
                        reading['reading_id'],
                        f"{reading['tunnel_length']:.2f}",
                        f"{reading['hor_dev_machine']:.2f}",
                        f"{reading['vert_dev_machine']:.2f}",
                        f"{reading['total_deviation_machine']:.2f}",
                        f"{reading['alignment_quality']:.3f}",
                        reading['quality_grade'],
                        reading['deviation_category'],
                        f"{reading['deviation_difference']:.2f}"
                    ]
                    writer.writerow(row)
            
            print(f"\nData exported to {filename}")
            return True
            
        except Exception as e:
            print(f"Error exporting data: {e}")
            return False
    
    def run_complete_analysis(self):
        """Run complete deviation analysis"""
        
        print("MTBM DEVIATION ANALYSIS TOOL")
        print("="*50)
        
        # Try to load real data first
        data_loaded = False
        
        # Try different possible data paths
        data_paths = [
            "AVN1200-ML/measure_protocol_original_.xls.csv",
            "measure_protocol_original_.xls.csv",
            "../measure_protocol_original_.xls.csv"
        ]
        
        for path in data_paths:
            if self.load_real_data(path):
                data_loaded = True
                break
        
        if not data_loaded:
            print("Real data not found, using sample data for demonstration")
            self.load_sample_data()
        
        # Process the data
        self.process_deviation_features()
        
        # Generate visualizations and analysis
        self.create_ascii_plots()
        self.generate_statistics_report()
        
        # Export results
        self.export_data_summary()
        
        print("\n" + "="*80)
        print("DEVIATION ANALYSIS COMPLETED")
        print("="*80)
        print("Generated:")
        print("• ASCII visualization plots")
        print("• Comprehensive statistics report")
        print("• CSV export with processed data")
        print("\nFor advanced graphical plots, install matplotlib and run deviation_visualization.py")


def main():
    """Main function"""
    analyzer = SimpleDeviationGraphGenerator()
    analyzer.run_complete_analysis()
    return analyzer


if __name__ == "__main__":
    result = main()