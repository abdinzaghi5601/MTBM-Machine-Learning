"""
Synthetic Tunneling Performance Data Generator

This module generates realistic synthetic data for MTBM (Micro-Tunneling Boring Machine)
operations, including geological conditions, machine parameters, and performance metrics.
"""

import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import random


class TunnelingDataGenerator:
    """Generate synthetic tunneling performance data for analytics and ML training."""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with random seed for reproducibility."""
        np.random.seed(seed)
        random.seed(seed)
        
        # Define geological conditions and their properties
        self.geological_types = {
            'soft_clay': {
                'ucs_range': (10, 50),  # kPa
                'abrasivity': (0.1, 0.3),
                'advance_rate_base': (20, 35),  # m/day
                'cutter_wear_rate': (0.005, 0.015),  # mm/hr
                'difficulty_factor': 0.3
            },
            'dense_sand': {
                'ucs_range': (50, 150),
                'abrasivity': (0.3, 0.6),
                'advance_rate_base': (15, 25),
                'cutter_wear_rate': (0.015, 0.035),
                'difficulty_factor': 0.6
            },
            'hard_rock': {
                'ucs_range': (1000, 5000),
                'abrasivity': (0.6, 1.0),
                'advance_rate_base': (8, 15),
                'cutter_wear_rate': (0.5, 1.2),
                'difficulty_factor': 1.0
            },
            'mixed_ground': {
                'ucs_range': (100, 800),
                'abrasivity': (0.4, 0.8),
                'advance_rate_base': (12, 20),
                'cutter_wear_rate': (0.2, 0.6),
                'difficulty_factor': 0.8
            }
        }
        
        # Machine specifications
        self.machine_specs = {
            'max_thrust': 2000,  # kN
            'max_torque': 500,   # kN·m
            'max_rpm': 15,       # rpm
            'max_advance_speed': 80,  # mm/min
            'cutter_diameter': 1200,  # mm
            'pipe_diameter': 1000     # mm
        }
    
    def generate_geological_profile(self, tunnel_length: float = 500) -> List[Dict]:
        """Generate geological profile along tunnel alignment."""
        profile = []
        current_chainage = 0
        
        while current_chainage < tunnel_length:
            # Random geological section length (20-100m)
            section_length = np.random.uniform(20, 100)
            section_length = min(section_length, tunnel_length - current_chainage)
            
            # Select geological type
            geo_type = np.random.choice(list(self.geological_types.keys()))
            geo_props = self.geological_types[geo_type]
            
            # Generate properties for this section
            section = {
                'start_chainage': current_chainage,
                'end_chainage': current_chainage + section_length,
                'length': section_length,
                'geological_type': geo_type,
                'ucs_strength': np.random.uniform(*geo_props['ucs_range']),
                'abrasivity_index': np.random.uniform(*geo_props['abrasivity']),
                'groundwater_level': np.random.uniform(5, 25),  # m below surface
                'difficulty_factor': geo_props['difficulty_factor']
            }
            
            profile.append(section)
            current_chainage += section_length
        
        return profile
    
    def generate_machine_parameters(self, geological_section: Dict, 
                                  base_timestamp: datetime) -> Dict:
        """Generate machine operating parameters based on geological conditions."""
        geo_type = geological_section['geological_type']
        geo_props = self.geological_types[geo_type]
        difficulty = geological_section['difficulty_factor']
        
        # Base parameters adjusted for geological conditions
        advance_speed = np.random.uniform(
            30 * (1 - difficulty * 0.6),  # Slower in difficult ground
            60 * (1 - difficulty * 0.3)
        )
        
        # Thrust force increases with difficulty
        thrust_force = np.random.uniform(
            800 + difficulty * 600,
            1500 + difficulty * 400
        )
        
        # RPM optimized for ground conditions
        revolution_rpm = np.random.uniform(
            6 + difficulty * 2,
            12 - difficulty * 2
        )
        
        # Working pressure based on ground conditions
        working_pressure = np.random.uniform(
            120 + difficulty * 50,
            200 + difficulty * 80
        )
        
        # Earth pressure (ground response)
        earth_pressure = working_pressure * np.random.uniform(0.7, 1.2)
        
        # Steering cylinder positions (random walk for realistic steering)
        steering_cylinders = {
            'top': np.random.uniform(-50, 50),
            'bottom': np.random.uniform(-50, 50),
            'left': np.random.uniform(-50, 50),
            'right': np.random.uniform(-50, 50)
        }
        
        # Deviation calculations (cumulative with some randomness)
        horizontal_deviation = np.random.normal(0, 5 + difficulty * 3)
        vertical_deviation = np.random.normal(0, 5 + difficulty * 3)
        
        return {
            'timestamp': base_timestamp,
            'chainage': geological_section['start_chainage'] + 
                       np.random.uniform(0, geological_section['length']),
            'geological_type': geo_type,
            'ucs_strength': geological_section['ucs_strength'],
            'abrasivity_index': geological_section['abrasivity_index'],
            'advance_speed': advance_speed,
            'revolution_rpm': revolution_rpm,
            'working_pressure': working_pressure,
            'total_thrust': thrust_force,
            'earth_pressure': earth_pressure,
            'steering_cylinder_top': steering_cylinders['top'],
            'steering_cylinder_bottom': steering_cylinders['bottom'],
            'steering_cylinder_left': steering_cylinders['left'],
            'steering_cylinder_right': steering_cylinders['right'],
            'horizontal_deviation_machine': horizontal_deviation,
            'vertical_deviation_machine': vertical_deviation,
            'horizontal_deviation_drill_head': horizontal_deviation + np.random.normal(0, 2),
            'vertical_deviation_drill_head': vertical_deviation + np.random.normal(0, 2),
            'cutter_wear_rate': np.random.uniform(*geo_props['cutter_wear_rate']),
            'grouting_volume': np.random.uniform(400, 600),  # L/m
            'grouting_pressure': np.random.uniform(6, 12),   # bar
        }
    
    def add_performance_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated performance metrics and engineered features."""
        # Total deviations
        df['total_deviation_machine'] = np.sqrt(
            df['horizontal_deviation_machine']**2 + 
            df['vertical_deviation_machine']**2
        )
        df['total_deviation_drill_head'] = np.sqrt(
            df['horizontal_deviation_drill_head']**2 + 
            df['vertical_deviation_drill_head']**2
        )
        
        # Alignment quality score
        df['alignment_quality'] = 1 / (1 + df['total_deviation_machine'])
        
        # Deviation difference
        df['deviation_difference'] = (
            df['total_deviation_drill_head'] - df['total_deviation_machine']
        )
        
        # Steering effort metrics
        steering_cols = ['steering_cylinder_top', 'steering_cylinder_bottom',
                        'steering_cylinder_left', 'steering_cylinder_right']
        df['steering_cylinder_range'] = df[steering_cols].max(axis=1) - df[steering_cols].min(axis=1)
        df['avg_cylinder_stroke'] = df[steering_cols].mean(axis=1)
        df['cylinder_variance'] = df[steering_cols].var(axis=1)
        
        # Performance efficiency metrics
        df['specific_energy'] = df['total_thrust'] / df['advance_speed']
        df['cutting_efficiency'] = df['advance_speed'] / df['revolution_rpm']
        df['pressure_efficiency'] = df['advance_speed'] / df['working_pressure']
        df['power_utilization'] = (df['total_thrust'] * df['advance_speed']) / 1000
        
        # Ground condition indicators
        df['ground_resistance'] = df['earth_pressure'] / df['advance_speed']
        df['penetration_rate'] = df['advance_speed'] / df['total_thrust']
        df['pressure_ratio'] = df['earth_pressure'] / df['working_pressure']
        
        # Operational efficiency
        df['operational_efficiency'] = df['advance_speed'] / df['specific_energy']
        df['system_stability'] = 1 / (1 + df['cylinder_variance'])
        
        # Quality classifications
        df['deviation_quality'] = pd.cut(
            df['alignment_quality'],
            bins=[0, 0.3, 0.5, 0.8, 1.0],
            labels=['Poor', 'Acceptable', 'Good', 'Excellent']
        )
        
        # Performance categories
        df['advance_rate_category'] = pd.cut(
            df['advance_speed'],
            bins=[0, 30, 45, 60, 100],
            labels=['Slow', 'Moderate', 'Good', 'Excellent']
        )
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['shift'] = df['hour'].apply(lambda x: 'night' if x < 6 or x > 18 else 'day')
        
        return df
    
    def generate_complete_dataset(self, 
                                tunnel_length: float = 500,
                                readings_per_meter: float = 2,
                                start_date: str = "2024-01-01") -> pd.DataFrame:
        """Generate complete synthetic tunneling dataset."""
        
        print(f"Generating synthetic tunneling data...")
        print(f"Tunnel length: {tunnel_length}m")
        print(f"Readings density: {readings_per_meter} per meter")
        
        # Generate geological profile
        geological_profile = self.generate_geological_profile(tunnel_length)
        
        # Calculate total number of readings
        total_readings = int(tunnel_length * readings_per_meter)
        
        # Generate time series
        start_datetime = datetime.strptime(start_date, "%Y-%m-%d")
        
        # Generate all data points
        data_points = []
        
        for i in range(total_readings):
            # Calculate current position
            current_chainage = (i / total_readings) * tunnel_length
            
            # Find corresponding geological section
            geological_section = None
            for section in geological_profile:
                if (section['start_chainage'] <= current_chainage <= 
                    section['end_chainage']):
                    geological_section = section
                    break
            
            if geological_section is None:
                geological_section = geological_profile[-1]  # Use last section as fallback
            
            # Calculate timestamp (assuming 8-hour shifts, 3 shifts per day)
            hours_elapsed = i * 0.5  # 30 minutes per reading average
            current_timestamp = start_datetime + timedelta(hours=hours_elapsed)
            
            # Generate machine parameters for this reading
            machine_data = self.generate_machine_parameters(
                geological_section, current_timestamp
            )
            machine_data['chainage'] = current_chainage
            
            data_points.append(machine_data)
        
        # Create DataFrame
        df = pd.DataFrame(data_points)
        
        # Add performance metrics and engineered features
        df = self.add_performance_metrics(df)
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        print(f"Generated {len(df)} data points")
        print(f"Geological sections: {len(geological_profile)}")
        print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        
        return df, geological_profile
    
    def save_dataset(self, df: pd.DataFrame, geological_profile: List[Dict],
                    output_dir: str = "data/synthetic"):
        """Save the generated dataset and metadata."""
        
        # Save main dataset
        df.to_csv(f"{output_dir}/tunneling_performance_data.csv", index=False)
        
        # Save geological profile
        with open(f"{output_dir}/tunnel_geological_profile.json", 'w') as f:
            json.dump(geological_profile, f, indent=2, default=str)
        
        # Generate dataset summary
        summary = {
            'generation_timestamp': datetime.now().isoformat(),
            'total_records': len(df),
            'tunnel_length': df['chainage'].max(),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'geological_sections': len(geological_profile),
            'geological_types': df['geological_type'].value_counts().to_dict(),
            'data_quality': {
                'missing_values': df.isnull().sum().sum(),
                'duplicate_records': df.duplicated().sum(),
                'data_completeness': f"{(1 - df.isnull().sum().sum() / df.size) * 100:.2f}%"
            },
            'performance_summary': {
                'avg_advance_speed': f"{df['advance_speed'].mean():.2f} mm/min",
                'avg_deviation': f"{df['total_deviation_machine'].mean():.2f} mm",
                'avg_alignment_quality': f"{df['alignment_quality'].mean():.3f}",
                'quality_distribution': df['deviation_quality'].value_counts().to_dict()
            }
        }
        
        with open(f"{output_dir}/dataset_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nDataset saved to {output_dir}/")
        print(f"Files created:")
        print(f"  - tunneling_performance_data.csv ({len(df)} records)")
        print(f"  - tunnel_geological_profile.json ({len(geological_profile)} sections)")
        print(f"  - dataset_summary.json (metadata)")


def main():
    """Generate and save synthetic tunneling performance dataset."""
    
    # Initialize generator
    generator = TunnelingDataGenerator(seed=42)
    
    # Generate dataset
    df, geological_profile = generator.generate_complete_dataset(
        tunnel_length=500,  # 500m tunnel
        readings_per_meter=2,  # 2 readings per meter
        start_date="2024-01-01"
    )
    
    # Save dataset
    generator.save_dataset(df, geological_profile)
    
    # Display sample statistics
    print("\n" + "="*60)
    print("DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Total records: {len(df):,}")
    print(f"Tunnel length: {df['chainage'].max():.1f}m")
    print(f"Time span: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
    print(f"\nGeological distribution:")
    for geo_type, count in df['geological_type'].value_counts().items():
        print(f"  {geo_type}: {count:,} records ({count/len(df)*100:.1f}%)")
    
    print(f"\nPerformance summary:")
    print(f"  Average advance speed: {df['advance_speed'].mean():.1f} mm/min")
    print(f"  Average deviation: {df['total_deviation_machine'].mean():.1f} mm")
    print(f"  Average alignment quality: {df['alignment_quality'].mean():.3f}")
    
    print(f"\nQuality distribution:")
    for quality, count in df['deviation_quality'].value_counts().items():
        print(f"  {quality}: {count:,} records ({count/len(df)*100:.1f}%)")


if __name__ == "__main__":
    main()