#!/usr/bin/env python3
"""
Unified MTBM Machine Learning Framework
======================================

Comprehensive ML system supporting all AVN protocols:
- AVN 800: Drive Protocol & Performance Optimization
- AVN 1200: Steering Accuracy & Deviation Control  
- AVN 2400: Advanced Measure Protocol
- AVN 3000: Predictive Planning & Resource Forecasting

This unified framework provides:
1. Multi-protocol data integration
2. Cross-protocol performance comparison
3. Unified predictive models
4. Comprehensive analytics dashboard
5. Real-time optimization recommendations

Author: ML for Tunneling Project
Date: November 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, classification_report
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class UnifiedMTBMFramework:
    """
    Unified Machine Learning framework for all MTBM protocols
    
    Supports AVN 800, 1200, 2400, and 3000 protocols with:
    - Cross-protocol data integration
    - Unified predictive models
    - Performance comparison analytics
    - Real-time optimization recommendations
    """
    
    def __init__(self):
        # Protocol-specific models
        self.models = {
            'avn800': {},  # Drive protocol models
            'avn1200': {}, # Steering accuracy models
            'avn2400': {}, # Advanced measure models
            'avn3000': {}  # Predictive planning models
        }
        
        # Data processing components
        self.scalers = {}
        self.encoders = {}
        self.feature_columns = {}
        
        # Performance tracking
        self.model_performance = {}
        self.cross_protocol_insights = {}
        
        # Protocol specifications
        self.protocol_specs = {
            'avn800': {
                'focus': 'Drive Performance & Equipment Optimization',
                'key_metrics': ['advance_rate', 'equipment_efficiency', 'downtime_prediction'],
                'data_frequency': 'continuous',
                'primary_sensors': ['thrust', 'torque', 'pressure', 'speed']
            },
            'avn1200': {
                'focus': 'Steering Accuracy & Tunnel Alignment',
                'key_metrics': ['deviation_control', 'steering_precision', 'alignment_quality'],
                'data_frequency': 'per_ring',
                'primary_sensors': ['deviation', 'cylinders', 'survey_data']
            },
            'avn2400': {
                'focus': 'Advanced Measurement & Quality Control',
                'key_metrics': ['measurement_accuracy', 'quality_indices', 'compliance_tracking'],
                'data_frequency': 'high_precision',
                'primary_sensors': ['precision_survey', 'quality_sensors', 'environmental']
            },
            'avn3000': {
                'focus': 'Predictive Planning & Resource Management',
                'key_metrics': ['geological_prediction', 'resource_forecasting', 'schedule_optimization'],
                'data_frequency': 'planning_intervals',
                'primary_sensors': ['geological', 'resource_usage', 'progress_tracking']
            }
        }
        
        # Ground condition classifications (unified across protocols)
        self.ground_types = {
            'soft_clay': {
                'characteristics': {'ucs': (10, 50), 'abrasivity': (0.1, 0.3), 'advance_rate': (25, 45)},
                'challenges': ['face_stability', 'ground_settlement'],
                'optimization': {'pressure_balance': 'critical', 'advance_speed': 'moderate'}
            },
            'dense_sand': {
                'characteristics': {'ucs': (50, 200), 'abrasivity': (0.3, 0.6), 'advance_rate': (15, 30)},
                'challenges': ['cutter_wear', 'ground_water'],
                'optimization': {'pressure_balance': 'important', 'advance_speed': 'controlled'}
            },
            'hard_rock': {
                'characteristics': {'ucs': (1000, 5000), 'abrasivity': (0.6, 1.0), 'advance_rate': (5, 20)},
                'challenges': ['high_wear', 'slow_progress', 'equipment_stress'],
                'optimization': {'pressure_balance': 'moderate', 'advance_speed': 'slow_steady'}
            },
            'mixed_ground': {
                'characteristics': {'ucs': (100, 1000), 'abrasivity': (0.4, 0.8), 'advance_rate': (10, 35)},
                'challenges': ['variable_conditions', 'parameter_adjustment'],
                'optimization': {'pressure_balance': 'adaptive', 'advance_speed': 'variable'}
            }
        }
    
    def generate_synthetic_data(self, protocol: str, n_samples: int = 500) -> pd.DataFrame:
        """
        Generate synthetic data for specific protocol
        """
        print(f"Generating synthetic data for {protocol.upper()} protocol...")
        
        np.random.seed(42)
        
        # Base data structure
        data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='30min'),
            'tunnel_length_m': np.linspace(0, 500, n_samples),
            'protocol': protocol,
            'protocol_focus': self.protocol_specs[protocol]['focus']
        }
        
        # Ground conditions simulation
        ground_zones = []
        for length in data['tunnel_length_m']:
            if length < 125:
                ground_zones.append('soft_clay')
            elif length < 250:
                ground_zones.append('dense_sand')
            elif length < 375:
                ground_zones.append('hard_rock')
            else:
                ground_zones.append('mixed_ground')
        
        data['ground_type'] = ground_zones
        
        # Generate protocol-specific operational data
        advance_speeds = []
        working_pressures = []
        
        for ground in ground_zones:
            ground_props = self.ground_types[ground]['characteristics']
            
            # Generate realistic values based on ground type
            advance_speed = np.random.uniform(*ground_props['advance_rate'])
            advance_speeds.append(advance_speed)
            
            # Working pressure based on ground difficulty
            if ground == 'soft_clay':
                pressure = np.random.uniform(120, 160)
            elif ground == 'dense_sand':
                pressure = np.random.uniform(160, 200)
            elif ground == 'hard_rock':
                pressure = np.random.uniform(200, 250)
            else:  # mixed_ground
                pressure = np.random.uniform(140, 220)
            
            working_pressures.append(pressure)
        
        data['advance_speed_mm_min'] = advance_speeds
        data['working_pressure_bar'] = working_pressures
        
        # Add protocol-specific features
        if protocol == 'avn1200':
            # Steering accuracy features
            data['hor_deviation_machine_mm'] = np.random.normal(0, 8, n_samples)
            data['vert_deviation_machine_mm'] = np.random.normal(0, 6, n_samples)
            data['alignment_quality'] = 1 / (1 + np.sqrt(np.array(data['hor_deviation_machine_mm'])**2 + 
                                                         np.array(data['vert_deviation_machine_mm'])**2) / 10)
        
        elif protocol == 'avn2400':
            # Advanced measurement features
            data['measurement_accuracy_mm'] = np.random.exponential(0.5, n_samples)
            data['quality_index'] = 100 - data['measurement_accuracy_mm'] * 20
            data['precision_score'] = np.random.beta(4, 2, n_samples)
        
        elif protocol == 'avn3000':
            # Predictive planning features
            data['resource_consumption'] = np.random.gamma(2, 50, n_samples)
            data['schedule_variance_hours'] = np.random.normal(0, 2, n_samples)
            data['planning_accuracy'] = np.random.uniform(0.7, 0.95, n_samples)
        
        # Common features for all protocols
        data['revolution_rpm'] = np.random.uniform(6, 12, n_samples)
        data['earth_pressure_bar'] = np.array(working_pressures) * np.random.uniform(0.8, 1.2, n_samples)
        
        return pd.DataFrame(data)
    
    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create unified feature set that works across all protocols
        """
        df_features = df.copy()
        
        # Universal features
        if 'advance_speed_mm_min' in df.columns:
            df_features['advance_rate_mh'] = df['advance_speed_mm_min'] * 60 / 1000  # m/hour
        
        if 'working_pressure_bar' in df.columns and 'advance_speed_mm_min' in df.columns:
            df_features['pressure_efficiency'] = df['advance_speed_mm_min'] / df['working_pressure_bar']
        
        # Ground condition encoding
        if 'ground_type' in df_features.columns:
            le = LabelEncoder()
            df_features['ground_type_encoded'] = le.fit_transform(df_features['ground_type'])
            self.encoders['ground_type'] = le
        
        # Temporal features
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['is_weekend'] = (df_features['timestamp'].dt.dayofweek >= 5).astype(int)
        
        # Performance score calculation
        if 'advance_speed_mm_min' in df.columns and 'ground_type' in df.columns:
            performance_scores = []
            for _, row in df_features.iterrows():
                ground = row['ground_type']
                speed = row['advance_speed_mm_min']
                expected_range = self.ground_types[ground]['characteristics']['advance_rate']
                normalized_score = (speed - expected_range[0]) / (expected_range[1] - expected_range[0])
                performance_scores.append(max(0, min(1, normalized_score)))
            
            df_features['performance_score'] = performance_scores
        
        return df_features
    
    def train_unified_models(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Train unified models across all protocols
        """
        print("Training Unified MTBM ML Models...")
        print("=" * 50)
        
        # Combine all datasets
        combined_df = pd.concat(datasets.values(), ignore_index=True)
        combined_df = self.engineer_features(combined_df)
        
        # Define prediction targets
        targets = {
            'advance_speed': 'advance_speed_mm_min',
            'performance_score': 'performance_score'
        }
        
        # Select features for modeling
        feature_columns = [
            'tunnel_length_m', 'ground_type_encoded', 'working_pressure_bar',
            'revolution_rpm', 'earth_pressure_bar', 'pressure_efficiency',
            'hour', 'day_of_week', 'is_weekend'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in combined_df.columns]
        
        results = {}
        
        for target_name, target_col in targets.items():
            if target_col and target_col in combined_df.columns:
                print(f"\nTraining models for {target_name}...")
                
                X = combined_df[available_features].fillna(0)
                y = combined_df[target_col]
                
                # Define models
                models = {
                    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
                    'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
                    'Ridge': Ridge(alpha=1.0)
                }
                
                target_results = {}
                
                for model_name, model in models.items():
                    # Train-test split
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Train model
                    model.fit(X_train, y_train)
                    
                    # Evaluate
                    y_pred = model.predict(X_test)
                    r2 = r2_score(y_test, y_pred)
                    
                    target_results[model_name] = {
                        'model': model,
                        'r2': r2
                    }
                    
                    print(f"  {model_name} R²: {r2:.3f}")
                
                results[target_name] = target_results
        
        # Store models and results
        self.models['unified'] = results
        self.feature_columns['unified'] = available_features
        
        return results
    
    def generate_cross_protocol_insights(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate insights comparing performance across protocols
        """
        print("\nGenerating Cross-Protocol Performance Insights...")
        print("=" * 50)
        
        insights = {}
        
        # Performance comparison by protocol
        protocol_performance = {}
        
        for protocol, df in datasets.items():
            df_features = self.engineer_features(df)
            
            if 'advance_speed_mm_min' in df_features.columns:
                avg_speed = df_features['advance_speed_mm_min'].mean()
                speed_std = df_features['advance_speed_mm_min'].std()
                
                protocol_performance[protocol] = {
                    'avg_advance_speed': avg_speed,
                    'speed_consistency': 1 / (1 + speed_std / avg_speed),
                    'data_points': len(df_features)
                }
        
        insights['protocol_performance'] = protocol_performance
        
        # Ground condition analysis
        ground_analysis = {}
        for protocol, df in datasets.items():
            if 'ground_type' in df.columns:
                ground_dist = df['ground_type'].value_counts(normalize=True)
                ground_analysis[protocol] = ground_dist.to_dict()
        
        insights['ground_distribution'] = ground_analysis
        
        # Print summary
        print("\nCROSS-PROTOCOL PERFORMANCE SUMMARY:")
        print("-" * 40)
        
        for protocol, perf in protocol_performance.items():
            print(f"{protocol.upper()}:")
            print(f"  Average Advance Speed: {perf['avg_advance_speed']:.1f} mm/min")
            print(f"  Consistency Score: {perf['speed_consistency']:.3f}")
            print(f"  Data Points: {perf['data_points']}")
            print()
        
        return insights
    
    def create_unified_dashboard(self, datasets: Dict[str, pd.DataFrame], save_plots: bool = True):
        """
        Create comprehensive dashboard showing all protocols
        """
        print("\nGenerating Unified MTBM Dashboard...")
        
        # Set up the plot style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create comprehensive dashboard
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Unified MTBM Performance Analytics Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Protocol Performance Comparison
        protocol_speeds = {}
        for protocol, df in datasets.items():
            if 'advance_speed_mm_min' in df.columns:
                protocol_speeds[protocol.upper()] = df['advance_speed_mm_min'].mean()
        
        if protocol_speeds:
            axes[0,0].bar(protocol_speeds.keys(), protocol_speeds.values(), 
                         color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[0,0].set_title('Average Advance Speed by Protocol')
            axes[0,0].set_ylabel('Speed (mm/min)')
            axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Ground Type Distribution
        combined_df = pd.concat(datasets.values(), ignore_index=True)
        if 'ground_type' in combined_df.columns:
            ground_counts = combined_df['ground_type'].value_counts()
            axes[0,1].pie(ground_counts.values, labels=ground_counts.index, autopct='%1.1f%%')
            axes[0,1].set_title('Ground Type Distribution (All Protocols)')
        
        # 3. Performance vs Ground Type
        if 'ground_type' in combined_df.columns and 'advance_speed_mm_min' in combined_df.columns:
            sns.boxplot(data=combined_df, x='ground_type', y='advance_speed_mm_min', ax=axes[1,0])
            axes[1,0].set_title('Performance by Ground Type')
            axes[1,0].tick_params(axis='x', rotation=45)
        
        # 4. Protocol Data Coverage
        protocol_coverage = {}
        for protocol, df in datasets.items():
            protocol_coverage[protocol.upper()] = len(df)
        
        axes[1,1].bar(protocol_coverage.keys(), protocol_coverage.values())
        axes[1,1].set_title('Data Coverage by Protocol')
        axes[1,1].set_ylabel('Number of Records')
        
        plt.tight_layout()
        
        if save_plots:
            try:
                plt.savefig('../../outputs/graphs/unified_mtbm_dashboard.png', dpi=300, bbox_inches='tight')
                print("Saved unified dashboard to outputs/graphs/unified_mtbm_dashboard.png")
            except Exception as e:
                print(f"Could not save plot: {e}")
        
        plt.show()
    
    def generate_comprehensive_report(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report for all protocols
        """
        print("\n" + "="*80)
        print("UNIFIED MTBM MACHINE LEARNING ANALYSIS REPORT")
        print("="*80)
        
        # Executive Summary
        total_records = sum(len(df) for df in datasets.values())
        protocols_analyzed = len(datasets)
        
        print(f"\nEXECUTIVE SUMMARY")
        print(f"Protocols Analyzed: {protocols_analyzed}")
        print(f"Total Data Points: {total_records:,}")
        print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        
        # Protocol-specific analysis
        print(f"\nPROTOCOL-SPECIFIC ANALYSIS")
        print("-" * 40)
        
        for protocol, df in datasets.items():
            print(f"\n{protocol.upper()} Protocol:")
            print(f"  Focus: {self.protocol_specs[protocol]['focus']}")
            print(f"  Records: {len(df):,}")
            
            if 'advance_speed_mm_min' in df.columns:
                avg_speed = df['advance_speed_mm_min'].mean()
                print(f"  Avg Advance Speed: {avg_speed:.1f} mm/min")
            
            if 'ground_type' in df.columns:
                dominant_ground = df['ground_type'].mode().iloc[0]
                print(f"  Dominant Ground: {dominant_ground}")
        
        # Cross-protocol insights
        insights = self.generate_cross_protocol_insights(datasets)
        
        # Recommendations
        print(f"\nSTRATEGIC RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = [
            "[PERFORMANCE] Implement cross-protocol parameter optimization",
            "[INTEGRATION] Deploy unified real-time monitoring system",
            "[ANALYTICS] Establish predictive maintenance protocols",
            "[AUTOMATION] Implement automated parameter adjustment systems"
        ]
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "="*80)
        
        return {'insights': insights, 'recommendations': recommendations}

def main():
    """
    Main execution function demonstrating unified MTBM framework
    """
    print("Unified MTBM Machine Learning Framework")
    print("======================================")
    
    # Initialize framework
    framework = UnifiedMTBMFramework()
    
    # Generate synthetic data for all protocols
    print("Generating synthetic data for all protocols...")
    
    datasets = {}
    for protocol in ['avn800', 'avn1200', 'avn2400', 'avn3000']:
        datasets[protocol] = framework.generate_synthetic_data(protocol, n_samples=500)
    
    # Train unified models
    model_results = framework.train_unified_models(datasets)
    
    # Generate cross-protocol insights
    insights = framework.generate_cross_protocol_insights(datasets)
    
    # Create unified dashboard
    framework.create_unified_dashboard(datasets)
    
    # Generate comprehensive report
    report = framework.generate_comprehensive_report(datasets)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()
