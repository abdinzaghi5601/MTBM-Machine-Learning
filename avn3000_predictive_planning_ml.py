#!/usr/bin/env python3
"""
AVN 3000 Predictive Planning ML Framework
==========================================

Advanced machine learning system for tunneling predictive planning with:
1. Look-ahead geological prediction (50-100m ahead)
2. Resource demand forecasting based on predicted geology
3. Monte Carlo simulation for schedule risk analysis

Data Structure from AVN 3000 Protocol:
- Date/Time stamps
- Survey tunnel length (m)
- Horizontal/Vertical deviation (mm)
- Survey pitch, reel, temperature
- Cylinder strokes and pressures
- Working pressure, revolution RPM
- Earth pressure measurements
- Advance speeds and forces

Author: ML for Tunneling Project
Date: 2025-08-09
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AVN3000PredictivePlanning:
    """
    Comprehensive predictive planning system for AVN 3000 MTBM operations
    """
    
    def __init__(self):
        self.geological_model = None
        self.resource_model = None
        self.schedule_model = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        
        # Ground condition classifications based on survey data patterns
        self.ground_classes = {
            'soft_clay': {'advance_speed_range': (30, 60), 'pressure_range': (20, 40)},
            'dense_sand': {'advance_speed_range': (15, 35), 'pressure_range': (40, 80)},
            'hard_rock': {'advance_speed_range': (5, 20), 'pressure_range': (80, 150)},
            'mixed_ground': {'advance_speed_range': (10, 40), 'pressure_range': (30, 100)}
        }
        
        # Resource consumption rates per ground type (per meter)
        self.resource_rates = {
            'soft_clay': {'bentonite_kg': 15, 'grout_liters': 180, 'cutter_wear_mm': 0.01},
            'dense_sand': {'bentonite_kg': 25, 'grout_liters': 220, 'cutter_wear_mm': 0.15},
            'hard_rock': {'bentonite_kg': 35, 'grout_liters': 280, 'cutter_wear_mm': 0.80},
            'mixed_ground': {'bentonite_kg': 22, 'grout_liters': 200, 'cutter_wear_mm': 0.25}
        }
        
    def load_avn3000_data(self, file_path):
        """
        Load and parse AVN 3000 measure protocol data
        Expected columns based on the protocol:
        1. Date, 2. Time, 3. Survey: Tunnel length, m
        4. Survey: Hor. deviation machine, mm, 5. Survey: Vert. deviation machine, mm
        9. Survey: Pitch, mm/m, 10. Survey: Reel, Degree
        11. Survey: Temperature in ELS/MWD, Degree
        17. Survey: Advance speed, mm/min, 18. Interjack: Force of TC and interjack, kN
        20. CW: Working pressure, bar, 21. CW: Revolution, rpm
        22. CW: Earth pressure 01 of excavation chamber, bar
        """
        try:
            # Read the data - assuming CSV format
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            else:
                # If it's an Excel file
                df = pd.read_excel(file_path)
            
            # Standardize column names
            column_mapping = {
                'Date': 'date',
                'Time': 'time', 
                'Survey: Tunnel length, m': 'tunnel_length_m',
                'Survey: Hor. deviation machine, mm': 'hor_deviation_mm',
                'Survey: Vert. deviation machine, mm': 'vert_deviation_mm',
                'Survey: Pitch, mm/m': 'pitch_mm_m',
                'Survey: Reel, Degree': 'reel_degree',
                'Survey: Temperature in ELS/MWD, Degree': 'temperature_degree',
                'Survey: Advance speed, mm/min': 'advance_speed_mm_min',
                'Interjack: Force of TC and interjack, kN': 'interjack_force_kn',
                'CW: Working pressure, bar': 'working_pressure_bar',
                'CW: Revolution, rpm': 'revolution_rpm',
                'CW: Earth pressure 01 of excavation chamber, bar': 'earth_pressure_bar'
            }
            
            # Rename columns if they exist
            for old_name, new_name in column_mapping.items():
                if old_name in df.columns:
                    df = df.rename(columns={old_name: new_name})
            
            # Create datetime column
            if 'date' in df.columns and 'time' in df.columns:
                df['datetime'] = pd.to_datetime(df['date'].astype(str) + ' ' + df['time'].astype(str), 
                                              errors='coerce')
            
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return self._generate_sample_data()
    
    def _generate_sample_data(self):
        """Generate sample AVN 3000 data for demonstration"""
        np.random.seed(42)
        n_samples = 500
        
        # Simulate 500m tunnel with varying conditions
        tunnel_length = np.linspace(0, 500, n_samples)
        
        # Create ground condition zones
        ground_zones = []
        for length in tunnel_length:
            if length < 150:
                ground_zones.append('soft_clay')
            elif length < 300:
                ground_zones.append('dense_sand')  
            elif length < 450:
                ground_zones.append('hard_rock')
            else:
                ground_zones.append('mixed_ground')
        
        data = {
            'datetime': pd.date_range('2023-01-01', periods=n_samples, freq='2H'),
            'tunnel_length_m': tunnel_length,
            'ground_type': ground_zones,
            'hor_deviation_mm': np.random.normal(0, 15, n_samples),
            'vert_deviation_mm': np.random.normal(0, 12, n_samples),
            'pitch_mm_m': np.random.uniform(-10, 10, n_samples),
            'reel_degree': np.random.uniform(0, 360, n_samples),
            'temperature_degree': np.random.uniform(15, 35, n_samples),
            'advance_speed_mm_min': [],
            'working_pressure_bar': [],
            'revolution_rpm': np.random.uniform(6, 12, n_samples),
            'earth_pressure_bar': [],
            'interjack_force_kn': np.random.uniform(500, 1500, n_samples)
        }
        
        # Generate realistic values based on ground type
        for i, ground in enumerate(ground_zones):
            speed_range = self.ground_classes[ground]['advance_speed_range']
            pressure_range = self.ground_classes[ground]['pressure_range']
            
            data['advance_speed_mm_min'].append(
                np.random.uniform(speed_range[0], speed_range[1]))
            data['working_pressure_bar'].append(
                np.random.uniform(pressure_range[0], pressure_range[1]))
            data['earth_pressure_bar'].append(
                np.random.uniform(pressure_range[0] * 0.8, pressure_range[1] * 1.2))
        
        return pd.DataFrame(data)
    
    def engineer_features(self, df):
        """Create advanced features for ML models"""
        df_features = df.copy()
        
        # Geological indicators
        df_features['total_deviation'] = np.sqrt(
            df['hor_deviation_mm']**2 + df['vert_deviation_mm']**2)
        df_features['deviation_rate'] = df_features['total_deviation'].rolling(5).std()
        
        # Performance indicators
        df_features['specific_energy'] = (
            df['working_pressure_bar'] * df['advance_speed_mm_min'] / 1000)
        df_features['advance_efficiency'] = df['advance_speed_mm_min'] / df['working_pressure_bar']
        
        # Trend features (look-back patterns)
        for window in [5, 10, 20]:
            df_features[f'advance_speed_ma_{window}'] = (
                df['advance_speed_mm_min'].rolling(window).mean())
            df_features[f'pressure_trend_{window}'] = (
                df['working_pressure_bar'].rolling(window).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == window else 0))
        
        # Ground stability indicators
        df_features['pressure_ratio'] = (
            df['earth_pressure_bar'] / df['working_pressure_bar'])
        df_features['force_per_speed'] = (
            df['interjack_force_kn'] / df['advance_speed_mm_min'])
        
        # Temporal features
        df_features['hour'] = df['datetime'].dt.hour
        df_features['day_of_week'] = df['datetime'].dt.dayofweek
        df_features['is_weekend'] = (df['datetime'].dt.dayofweek >= 5).astype(int)
        
        return df_features
    
    def classify_ground_conditions(self, df):
        """Classify ground conditions from operational parameters"""
        ground_conditions = []
        
        for _, row in df.iterrows():
            advance_speed = row['advance_speed_mm_min']
            pressure = row['working_pressure_bar']
            
            # Classification logic based on operational patterns
            if advance_speed > 40 and pressure < 50:
                ground_conditions.append('soft_clay')
            elif advance_speed < 20 and pressure > 80:
                ground_conditions.append('hard_rock')
            elif 20 <= advance_speed <= 40 and 50 <= pressure <= 80:
                ground_conditions.append('dense_sand')
            else:
                ground_conditions.append('mixed_ground')
        
        return ground_conditions
    
    def train_geological_prediction_model(self, df):
        """Train look-ahead geological prediction model"""
        print("Training Geological Prediction Model...")
        
        df_features = self.engineer_features(df)
        
        # Classify current ground conditions
        df_features['ground_type'] = self.classify_ground_conditions(df_features)
        
        # Create look-ahead targets (predict conditions 50-100m ahead)
        look_ahead_distances = [50, 75, 100]
        
        models = {}
        for distance in look_ahead_distances:
            # Calculate approximate rows ahead (assuming ~1m per row)
            rows_ahead = int(distance)
            
            if len(df_features) > rows_ahead:
                # Prepare features and targets
                feature_cols = [
                    'tunnel_length_m', 'advance_speed_mm_min', 'working_pressure_bar',
                    'earth_pressure_bar', 'specific_energy', 'advance_efficiency',
                    'total_deviation', 'pressure_ratio', 'temperature_degree'
                ]
                
                X = df_features[feature_cols].fillna(0)
                y = self.label_encoder.fit_transform(
                    df_features['ground_type'].shift(-rows_ahead).fillna('mixed_ground'))
                
                # Remove NaN rows
                valid_indices = ~pd.isna(y)
                X = X[valid_indices]
                y = y[valid_indices]
                
                if len(X) > 10:  # Minimum data requirement
                    # Train model
                    model = RandomForestRegressor(n_estimators=100, random_state=42)
                    model.fit(X, y)
                    models[distance] = model
                    
                    # Evaluate model
                    score = model.score(X, y)
                    print(f"  {distance}m ahead model R²: {score:.3f}")
        
        self.geological_model = models
        self.feature_columns = feature_cols
        return models
    
    def predict_geological_conditions(self, df, distances=[50, 75, 100]):
        """Predict geological conditions ahead"""
        if not self.geological_model:
            print("Geological model not trained. Training now...")
            self.train_geological_prediction_model(df)
        
        df_features = self.engineer_features(df)
        X = df_features[self.feature_columns].fillna(0)
        
        predictions = {}
        confidence_scores = {}
        
        for distance in distances:
            if distance in self.geological_model:
                model = self.geological_model[distance]
                
                # Get predictions
                pred_encoded = model.predict(X)
                
                # Convert back to ground types
                pred_classes = []
                for pred in pred_encoded:
                    pred_int = int(round(pred))
                    if 0 <= pred_int < len(self.label_encoder.classes_):
                        pred_classes.append(self.label_encoder.classes_[pred_int])
                    else:
                        pred_classes.append('mixed_ground')
                
                predictions[distance] = pred_classes
                
                # Calculate confidence (using prediction variance from trees)
                if hasattr(model, 'estimators_'):
                    tree_predictions = np.array([
                        tree.predict(X) for tree in model.estimators_])
                    confidence_scores[distance] = 1 - np.std(tree_predictions, axis=0)
                else:
                    confidence_scores[distance] = np.ones(len(X)) * 0.8
        
        return predictions, confidence_scores
    
    def train_resource_forecasting_model(self, df):
        """Train resource demand forecasting model"""
        print("Training Resource Forecasting Model...")
        
        df_features = self.engineer_features(df)
        df_features['ground_type'] = self.classify_ground_conditions(df_features)
        
        # Calculate resource consumption based on ground type and distance
        resource_consumption = {
            'bentonite_kg': [],
            'grout_liters': [],
            'cutter_wear_mm': [],
            'advance_time_hours': []
        }
        
        for _, row in df_features.iterrows():
            ground = row['ground_type']
            rates = self.resource_rates[ground]
            
            # Calculate consumption per meter
            resource_consumption['bentonite_kg'].append(rates['bentonite_kg'])
            resource_consumption['grout_liters'].append(rates['grout_liters'])
            resource_consumption['cutter_wear_mm'].append(rates['cutter_wear_mm'])
            
            # Calculate time based on advance speed
            time_hours = 60 / row['advance_speed_mm_min'] if row['advance_speed_mm_min'] > 0 else 2
            resource_consumption['advance_time_hours'].append(time_hours)
        
        # Add to dataframe
        for resource, values in resource_consumption.items():
            df_features[resource] = values
        
        # Train models for each resource type
        feature_cols = [
            'tunnel_length_m', 'advance_speed_mm_min', 'working_pressure_bar',
            'earth_pressure_bar', 'specific_energy', 'total_deviation'
        ]
        
        resource_models = {}
        
        for resource in ['bentonite_kg', 'grout_liters', 'cutter_wear_mm', 'advance_time_hours']:
            X = df_features[feature_cols].fillna(0)
            y = df_features[resource]
            
            # Train ensemble model
            models = {
                'rf': RandomForestRegressor(n_estimators=100, random_state=42),
                'gb': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'ridge': Ridge(alpha=1.0)
            }
            
            trained_models = {}
            for name, model in models.items():
                model.fit(X, y)
                trained_models[name] = model
                
                # Evaluate
                score = model.score(X, y)
                print(f"  {resource} - {name} R²: {score:.3f}")
            
            resource_models[resource] = trained_models
        
        self.resource_model = resource_models
        self.resource_feature_columns = feature_cols
        return resource_models
    
    def forecast_resource_demands(self, df, predicted_geology, distances=[50, 75, 100]):
        """Forecast resource demands based on predicted geology"""
        if not self.resource_model:
            print("Resource model not trained. Training now...")
            self.train_resource_forecasting_model(df)
        
        df_features = self.engineer_features(df)
        X = df_features[self.resource_feature_columns].fillna(0)
        
        forecasts = {}
        
        for distance in distances:
            if distance in predicted_geology:
                geology_predictions = predicted_geology[distance]
                
                # Forecast resources for each predicted ground type
                distance_forecasts = {}
                
                for resource_type in ['bentonite_kg', 'grout_liters', 'cutter_wear_mm', 'advance_time_hours']:
                    models = self.resource_model[resource_type]
                    
                    # Use ensemble prediction (average of models)
                    predictions = []
                    for model_name, model in models.items():
                        pred = model.predict(X)
                        predictions.append(pred)
                    
                    # Average predictions
                    ensemble_pred = np.mean(predictions, axis=0)
                    
                    # Adjust based on predicted geology
                    adjusted_pred = []
                    for i, geology in enumerate(geology_predictions):
                        if i < len(ensemble_pred):
                            base_pred = ensemble_pred[i]
                            # Apply geology-specific multipliers
                            if geology == 'hard_rock':
                                multiplier = 1.5
                            elif geology == 'soft_clay':
                                multiplier = 0.7
                            elif geology == 'dense_sand':
                                multiplier = 1.2
                            else:  # mixed_ground
                                multiplier = 1.0
                            
                            adjusted_pred.append(base_pred * multiplier)
                    
                    distance_forecasts[resource_type] = adjusted_pred
                
                forecasts[distance] = distance_forecasts
        
        return forecasts
    
    def monte_carlo_schedule_analysis(self, df, n_simulations=1000):
        """Perform Monte Carlo simulation for schedule risk analysis"""
        print("Running Monte Carlo Schedule Risk Analysis...")
        
        df_features = self.engineer_features(df)
        
        # Analyze historical performance variations
        advance_speeds = df_features['advance_speed_mm_min'].dropna()
        speed_mean = advance_speeds.mean()
        speed_std = advance_speeds.std()
        
        # Analyze downtime patterns
        # Estimate downtime from gaps in data or low advance speeds
        downtime_rates = []
        for i in range(1, len(df_features)):
            time_diff = (df_features.iloc[i]['datetime'] - 
                        df_features.iloc[i-1]['datetime']).total_seconds() / 3600
            if time_diff > 4:  # More than 4 hours gap indicates downtime
                downtime_rates.append(time_diff)
        
        if downtime_rates:
            downtime_mean = np.mean(downtime_rates)
            downtime_std = np.std(downtime_rates)
        else:
            downtime_mean, downtime_std = 8, 4  # Default values
        
        # Ground condition impact factors
        ground_impact = {
            'soft_clay': {'speed_factor': 1.3, 'downtime_factor': 0.7},
            'dense_sand': {'speed_factor': 1.0, 'downtime_factor': 1.0},
            'hard_rock': {'speed_factor': 0.6, 'downtime_factor': 1.8},
            'mixed_ground': {'speed_factor': 0.9, 'downtime_factor': 1.2}
        }
        
        # Monte Carlo simulation
        completion_times = []
        
        for sim in range(n_simulations):
            total_time = 0
            remaining_length = df_features['tunnel_length_m'].max()
            current_pos = df_features['tunnel_length_m'].min()
            
            while current_pos < remaining_length:
                # Determine ground conditions (simplified)
                if current_pos < 150:
                    ground = 'soft_clay'
                elif current_pos < 300:
                    ground = 'dense_sand'
                elif current_pos < 450:
                    ground = 'hard_rock'
                else:
                    ground = 'mixed_ground'
                
                # Sample advance speed with ground condition impact
                base_speed = np.random.normal(speed_mean, speed_std)
                actual_speed = base_speed * ground_impact[ground]['speed_factor']
                actual_speed = max(actual_speed, 5)  # Minimum speed
                
                # Calculate time for next 10m segment
                segment_length = min(10, remaining_length - current_pos)
                segment_time = (segment_length * 1000) / actual_speed / 60  # hours
                
                # Add random downtime
                if np.random.random() < 0.3:  # 30% chance of downtime per segment
                    downtime = np.random.normal(downtime_mean, downtime_std)
                    downtime = max(downtime, 0) * ground_impact[ground]['downtime_factor']
                    segment_time += downtime
                
                total_time += segment_time
                current_pos += segment_length
            
            completion_times.append(total_time)
        
        # Analysis results
        completion_times = np.array(completion_times)
        
        results = {
            'mean_completion_time': np.mean(completion_times),
            'std_completion_time': np.std(completion_times),
            'percentile_50': np.percentile(completion_times, 50),
            'percentile_80': np.percentile(completion_times, 80),
            'percentile_95': np.percentile(completion_times, 95),
            'risk_of_delay': np.sum(completion_times > np.percentile(completion_times, 80)) / len(completion_times),
            'completion_times': completion_times
        }
        
        print(f"Schedule Analysis Results:")
        print(f"  Mean completion time: {results['mean_completion_time']:.1f} hours")
        print(f"  50th percentile: {results['percentile_50']:.1f} hours")
        print(f"  80th percentile: {results['percentile_80']:.1f} hours")
        print(f"  95th percentile: {results['percentile_95']:.1f} hours")
        print(f"  Risk of significant delay (>80th percentile): {results['risk_of_delay']:.1%}")
        
        return results
    
    def generate_predictive_planning_report(self, df):
        """Generate comprehensive predictive planning report"""
        print("Generating Predictive Planning Report...")
        print("=" * 60)
        
        # 1. Data Overview
        print("1. DATA OVERVIEW")
        print(f"   Tunnel length analyzed: {df['tunnel_length_m'].min():.1f}m to {df['tunnel_length_m'].max():.1f}m")
        print(f"   Total data points: {len(df)}")
        print(f"   Date range: {df['datetime'].min()} to {df['datetime'].max()}")
        print()
        
        # 2. Geological Predictions
        print("2. GEOLOGICAL LOOK-AHEAD PREDICTIONS")
        predictions, confidence = self.predict_geological_conditions(df)
        
        for distance in [50, 75, 100]:
            if distance in predictions:
                pred_counts = pd.Series(predictions[distance]).value_counts()
                avg_confidence = np.mean(confidence[distance])
                
                print(f"   {distance}m Ahead Predictions:")
                for ground_type, count in pred_counts.items():
                    percentage = count / len(predictions[distance]) * 100
                    print(f"     {ground_type}: {percentage:.1f}%")
                print(f"     Average Confidence: {avg_confidence:.2f}")
                print()
        
        # 3. Resource Forecasting
        print("3. RESOURCE DEMAND FORECASTS")
        resource_forecasts = self.forecast_resource_demands(df, predictions)
        
        for distance in [50, 75, 100]:
            if distance in resource_forecasts:
                forecasts = resource_forecasts[distance]
                print(f"   {distance}m Ahead Resource Needs:")
                
                for resource_type in ['bentonite_kg', 'grout_liters', 'cutter_wear_mm']:
                    if resource_type in forecasts:
                        total_demand = np.sum(forecasts[resource_type])
                        avg_rate = np.mean(forecasts[resource_type])
                        print(f"     {resource_type}: {total_demand:.1f} total, {avg_rate:.2f} per meter")
                print()
        
        # 4. Schedule Risk Analysis
        print("4. SCHEDULE RISK ANALYSIS")
        schedule_results = self.monte_carlo_schedule_analysis(df)
        
        print(f"   Expected completion: {schedule_results['mean_completion_time']:.1f} hours")
        print(f"   Conservative estimate (80%): {schedule_results['percentile_80']:.1f} hours")
        print(f"   Worst-case scenario (95%): {schedule_results['percentile_95']:.1f} hours")
        print(f"   Schedule risk: {schedule_results['risk_of_delay']:.1%} probability of delays")
        print()
        
        # 5. Recommendations
        print("5. STRATEGIC RECOMMENDATIONS")
        
        # Analyze dominant ground conditions
        all_predictions = []
        for distance in predictions.values():
            all_predictions.extend(distance)
        
        pred_summary = pd.Series(all_predictions).value_counts(normalize=True)
        
        if 'hard_rock' in pred_summary and pred_summary['hard_rock'] > 0.3:
            print("   ⚠️  HIGH HARD ROCK CONTENT DETECTED")
            print("      - Increase cutter tool inventory by 50%")
            print("      - Schedule additional maintenance windows")
            print("      - Consider backup equipment deployment")
            print()
        
        if 'soft_clay' in pred_summary and pred_summary['soft_clay'] > 0.4:
            print("   💧 SOFT GROUND CONDITIONS AHEAD")
            print("      - Increase bentonite supply")
            print("      - Monitor ground stability closely")
            print("      - Prepare ground conditioning systems")
            print()
        
        print("   📊 OPTIMIZATION OPPORTUNITIES")
        avg_speed = df['advance_speed_mm_min'].mean()
        if avg_speed < 25:
            print("      - Current advance rate below optimal")
            print("      - Consider parameter optimization")
        
        print("      - Implement real-time geological monitoring")
        print("      - Establish predictive maintenance schedule")
        print("      - Deploy automated resource tracking")
        
        print("=" * 60)
        
        return {
            'geological_predictions': predictions,
            'resource_forecasts': resource_forecasts,
            'schedule_analysis': schedule_results,
            'confidence_scores': confidence
        }

def main():
    """Main execution function"""
    print("AVN 3000 Predictive Planning ML Framework")
    print("=========================================")
    
    # Initialize system
    planning_system = AVN3000PredictivePlanning()
    
    # For demonstration, we'll use sample data
    # In real implementation, load from: planning_system.load_avn3000_data("path/to/data.csv")
    print("Loading demonstration data...")
    df = planning_system._generate_sample_data()
    
    # Train all models and generate comprehensive report
    results = planning_system.generate_predictive_planning_report(df)
    
    # Save results
    print("\nSaving analysis results...")
    
    # Save predictions to CSV
    predictions_df = pd.DataFrame()
    for distance, preds in results['geological_predictions'].items():
        predictions_df[f'geology_{distance}m'] = preds[:len(df)]
        predictions_df[f'confidence_{distance}m'] = results['confidence_scores'][distance][:len(df)]
    
    predictions_df['tunnel_position'] = df['tunnel_length_m']
    predictions_df.to_csv('avn3000_geological_predictions.csv', index=False)
    
    # Save resource forecasts
    resource_df = pd.DataFrame()
    for distance, forecasts in results['resource_forecasts'].items():
        for resource_type, values in forecasts.items():
            resource_df[f'{resource_type}_{distance}m'] = values[:len(df)]
    
    resource_df.to_csv('avn3000_resource_forecasts.csv', index=False)
    
    print("Analysis complete! Files saved:")
    print("- avn3000_geological_predictions.csv")
    print("- avn3000_resource_forecasts.csv")

if __name__ == "__main__":
    main()