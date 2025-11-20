#!/usr/bin/env python3
"""
MTBM Cutter Wear Prediction & Geological Correlation ML Framework
Advanced machine learning system for predicting cutter wear patterns and optimizing boring parameters
based on geological conditions and operational data.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge, ElasticNet
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class CutterWearPredictionML:
    """
    Comprehensive ML framework for cutter wear prediction and geological correlation analysis
    
    Features:
    1. Cutter wear pattern prediction based on geological conditions
    2. Geological correlation analysis for parameter optimization  
    3. Boring parameter recommendation system
    4. Real-time wear monitoring and alerts
    5. Maintenance scheduling optimization
    """
    
    def __init__(self):
        # Model components
        self.wear_prediction_model = None
        self.geological_classifier = None
        self.parameter_optimizer = None
        self.maintenance_scheduler = None
        
        # Data processing components
        self.scaler = StandardScaler()
        self.geological_encoder = LabelEncoder()
        self.parameter_scaler = MinMaxScaler()
        
        # Feature tracking
        self.wear_features = []
        self.geological_features = []
        self.parameter_features = []
        
        # Model performance tracking
        self.model_performance = {}
        
        # Geological classification mapping
        self.geological_classes = {
            'soft_clay': 0, 'hard_clay': 1, 'sandy_clay': 2,
            'loose_sand': 3, 'dense_sand': 4, 'gravel': 5,
            'soft_rock': 6, 'medium_rock': 7, 'hard_rock': 8,
            'mixed_ground': 9, 'weathered_rock': 10
        }
        
        # Cutter wear thresholds (mm)
        self.wear_thresholds = {
            'new': 0.0,
            'light_wear': 2.0,
            'moderate_wear': 5.0,
            'heavy_wear': 8.0,
            'replacement_needed': 12.0
        }
    
    def load_cutter_wear_data(self, csv_path: str = None) -> pd.DataFrame:
        """Load cutter wear and operational data"""
        
        if csv_path is None:
            # Generate comprehensive sample data for demonstration
            df = self.generate_sample_wear_data()
            print("Using generated sample cutter wear data")
        else:
            # Load real data
            df = pd.read_csv(csv_path)
            print(f"Loaded cutter wear data from {csv_path}")
        
        return df
    
    def generate_sample_wear_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """Generate realistic sample data for cutter wear analysis"""
        
        np.random.seed(42)
        
        # Simulate tunneling operations over time
        data = []
        
        for i in range(n_samples):
            # Progressive tunnel advance
            tunnel_distance = i * 0.5  # 50cm per reading
            operating_hours = i * 0.25  # 15 minutes per reading
            
            # Geological conditions (changes throughout tunnel)
            geological_zones = [
                ('soft_clay', 0, 200), ('sandy_clay', 200, 350), 
                ('dense_sand', 350, 450), ('gravel', 450, 600),
                ('soft_rock', 600, 750), ('medium_rock', 750, 900),
                ('hard_rock', 900, 1000), ('mixed_ground', 1000, 1200)
            ]
            
            current_geology = 'soft_clay'  # Default
            for geo_type, start, end in geological_zones:
                if start <= tunnel_distance < end:
                    current_geology = geo_type
                    break
            
            # Geological properties based on type
            geological_properties = self.get_geological_properties(current_geology)
            
            # Machine operational parameters
            advance_speed = np.random.normal(45, 8)  # mm/min
            revolution_rpm = np.random.normal(8.5, 1.2)
            working_pressure = np.random.normal(180, 25)
            earth_pressure = geological_properties['base_pressure'] + np.random.normal(0, 10)
            total_force = geological_properties['base_force'] + np.random.normal(0, 50)
            
            # Cutter-specific parameters
            num_cutters = 24  # Typical for MTBM
            
            # Calculate wear factors
            wear_rate = self.calculate_wear_rate(
                current_geology, advance_speed, revolution_rpm, 
                earth_pressure, total_force, operating_hours
            )
            
            # Individual cutter wear (varies by position)
            cutter_positions = ['face_center', 'face_outer', 'gauge', 'back_reamer']
            position_multipliers = {'face_center': 1.2, 'face_outer': 1.0, 'gauge': 1.5, 'back_reamer': 0.8}
            
            for cutter_id in range(num_cutters):
                position = cutter_positions[cutter_id % 4]
                position_multiplier = position_multipliers[position]
                
                # Cumulative wear calculation
                base_wear = wear_rate * operating_hours * position_multiplier
                wear_variation = np.random.normal(1.0, 0.15)  # Individual cutter variation
                total_wear = base_wear * wear_variation
                
                # Add some randomness for realistic patterns
                total_wear += np.random.normal(0, 0.2)
                total_wear = max(0, total_wear)  # No negative wear
                
                # Determine wear condition
                wear_condition = self.classify_wear_condition(total_wear)
                
                # Predicted remaining life
                remaining_life = max(0, (self.wear_thresholds['replacement_needed'] - total_wear) / 
                                   (wear_rate * position_multiplier + 0.001))
                
                data.append({
                    'reading_id': i + 1,
                    'cutter_id': cutter_id + 1,
                    'tunnel_distance': tunnel_distance,
                    'operating_hours': operating_hours,
                    'geological_type': current_geology,
                    'cutter_position': position,
                    
                    # Geological properties
                    'ucs_strength': geological_properties['ucs'],
                    'abrasivity_index': geological_properties['abrasivity'],
                    'hardness_shore': geological_properties['hardness'],
                    'quartz_content': geological_properties['quartz'],
                    
                    # Machine parameters
                    'advance_speed': advance_speed,
                    'revolution_rpm': revolution_rpm,
                    'working_pressure': working_pressure,
                    'earth_pressure': earth_pressure,
                    'total_force': total_force,
                    
                    # Cutter wear data
                    'total_wear': total_wear,
                    'wear_rate': wear_rate * position_multiplier,
                    'wear_condition': wear_condition,
                    'remaining_life_hours': remaining_life,
                    
                    # Target variables for ML
                    'next_inspection_wear': total_wear + (wear_rate * position_multiplier * 24),  # 24 hours ahead
                    'replacement_needed_in_hours': remaining_life
                })
        
        return pd.DataFrame(data)
    
    def get_geological_properties(self, geology_type: str) -> Dict:
        """Get geological properties for each ground type"""
        
        properties = {
            'soft_clay': {
                'ucs': np.random.normal(0.5, 0.2), 'abrasivity': np.random.normal(2, 0.5),
                'hardness': np.random.normal(20, 5), 'quartz': np.random.normal(10, 3),
                'base_pressure': 80, 'base_force': 400
            },
            'hard_clay': {
                'ucs': np.random.normal(2, 0.5), 'abrasivity': np.random.normal(3, 0.5),
                'hardness': np.random.normal(35, 8), 'quartz': np.random.normal(15, 4),
                'base_pressure': 120, 'base_force': 600
            },
            'sandy_clay': {
                'ucs': np.random.normal(1.5, 0.4), 'abrasivity': np.random.normal(4, 0.8),
                'hardness': np.random.normal(30, 6), 'quartz': np.random.normal(25, 6),
                'base_pressure': 100, 'base_force': 550
            },
            'loose_sand': {
                'ucs': np.random.normal(0.1, 0.05), 'abrasivity': np.random.normal(3.5, 0.6),
                'hardness': np.random.normal(15, 4), 'quartz': np.random.normal(35, 8),
                'base_pressure': 60, 'base_force': 350
            },
            'dense_sand': {
                'ucs': np.random.normal(0.8, 0.3), 'abrasivity': np.random.normal(4.5, 0.7),
                'hardness': np.random.normal(40, 8), 'quartz': np.random.normal(45, 10),
                'base_pressure': 140, 'base_force': 700
            },
            'gravel': {
                'ucs': np.random.normal(3, 0.8), 'abrasivity': np.random.normal(5.5, 1.0),
                'hardness': np.random.normal(50, 10), 'quartz': np.random.normal(30, 8),
                'base_pressure': 160, 'base_force': 800
            },
            'soft_rock': {
                'ucs': np.random.normal(15, 4), 'abrasivity': np.random.normal(4, 0.8),
                'hardness': np.random.normal(45, 10), 'quartz': np.random.normal(20, 6),
                'base_pressure': 180, 'base_force': 900
            },
            'medium_rock': {
                'ucs': np.random.normal(50, 12), 'abrasivity': np.random.normal(5, 1.0),
                'hardness': np.random.normal(60, 12), 'quartz': np.random.normal(25, 8),
                'base_pressure': 220, 'base_force': 1100
            },
            'hard_rock': {
                'ucs': np.random.normal(120, 30), 'abrasivity': np.random.normal(6.5, 1.2),
                'hardness': np.random.normal(80, 15), 'quartz': np.random.normal(40, 12),
                'base_pressure': 280, 'base_force': 1400
            },
            'mixed_ground': {
                'ucs': np.random.normal(25, 15), 'abrasivity': np.random.normal(4.5, 1.5),
                'hardness': np.random.normal(45, 20), 'quartz': np.random.normal(30, 15),
                'base_pressure': 150, 'base_force': 750
            },
            'weathered_rock': {
                'ucs': np.random.normal(8, 3), 'abrasivity': np.random.normal(3.5, 0.8),
                'hardness': np.random.normal(35, 8), 'quartz': np.random.normal(18, 5),
                'base_pressure': 130, 'base_force': 650
            }
        }
        
        return properties.get(geology_type, properties['soft_clay'])
    
    def calculate_wear_rate(self, geology: str, advance_speed: float, rpm: float, 
                          earth_pressure: float, total_force: float, hours: float) -> float:
        """Calculate cutter wear rate based on operational and geological conditions"""
        
        # Base wear rates by geology (mm/hour)
        base_rates = {
            'soft_clay': 0.02, 'hard_clay': 0.05, 'sandy_clay': 0.08,
            'loose_sand': 0.06, 'dense_sand': 0.12, 'gravel': 0.18,
            'soft_rock': 0.15, 'medium_rock': 0.25, 'hard_rock': 0.45,
            'mixed_ground': 0.20, 'weathered_rock': 0.10
        }
        
        base_rate = base_rates.get(geology, 0.10)
        
        # Operational multipliers
        speed_factor = (advance_speed / 45) ** 0.5  # Square root relationship
        rpm_factor = (rpm / 8.5) ** 0.3  # Moderate influence
        pressure_factor = (earth_pressure / 120) ** 0.7  # Strong influence
        force_factor = (total_force / 800) ** 0.6  # Strong influence
        
        # Time-dependent wear acceleration
        time_factor = 1 + (hours / 1000) * 0.1  # Slight acceleration over time
        
        wear_rate = base_rate * speed_factor * rpm_factor * pressure_factor * force_factor * time_factor
        
        return max(0.001, wear_rate)  # Minimum wear rate
    
    def classify_wear_condition(self, wear_amount: float) -> str:
        """Classify cutter wear condition based on wear amount"""
        
        if wear_amount < self.wear_thresholds['light_wear']:
            return 'new'
        elif wear_amount < self.wear_thresholds['moderate_wear']:
            return 'light_wear'
        elif wear_amount < self.wear_thresholds['heavy_wear']:
            return 'moderate_wear'
        elif wear_amount < self.wear_thresholds['replacement_needed']:
            return 'heavy_wear'
        else:
            return 'replacement_needed'
    
    def engineer_wear_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineer features for cutter wear prediction"""
        
        print("Engineering cutter wear prediction features...")
        
        # 1. GEOLOGICAL FEATURES
        df['geological_risk_score'] = (
            df['ucs_strength'] * 0.3 + 
            df['abrasivity_index'] * 0.4 + 
            df['hardness_shore'] * 0.2 + 
            df['quartz_content'] * 0.1
        )
        
        # Geological severity classification
        df['geological_severity'] = pd.cut(
            df['geological_risk_score'],
            bins=[-np.inf, 20, 50, 100, np.inf],
            labels=['low', 'medium', 'high', 'extreme']
        )
        
        # 2. OPERATIONAL INTENSITY FEATURES
        df['cutting_intensity'] = df['advance_speed'] * df['revolution_rpm'] / 100
        df['pressure_stress'] = df['earth_pressure'] / df['working_pressure']
        df['force_per_revolution'] = df['total_force'] / (df['revolution_rpm'] + 0.1)
        df['specific_force'] = df['total_force'] / (df['advance_speed'] + 0.1)
        
        # 3. CUMULATIVE WEAR FACTORS
        df['cumulative_cutting_distance'] = df.groupby('cutter_id')['advance_speed'].cumsum() * 0.25 / 1000  # km
        df['cumulative_revolutions'] = df.groupby('cutter_id')['revolution_rpm'].cumsum() * 0.25  # total revolutions
        df['cumulative_force_exposure'] = df.groupby('cutter_id')['total_force'].cumsum() * 0.25  # force-hours
        
        # 4. POSITION-BASED FEATURES
        position_wear_factors = {'face_center': 1.2, 'face_outer': 1.0, 'gauge': 1.5, 'back_reamer': 0.8}
        df['position_wear_factor'] = df['cutter_position'].map(position_wear_factors)
        
        # 5. INTERACTION FEATURES
        df['geology_force_interaction'] = df['geological_risk_score'] * df['specific_force']
        df['speed_hardness_interaction'] = df['advance_speed'] * df['hardness_shore']
        df['abrasive_cutting_interaction'] = df['abrasivity_index'] * df['cutting_intensity']
        
        # 6. TEMPORAL FEATURES
        df['operating_days'] = df['operating_hours'] / 24
        df['wear_acceleration'] = df['total_wear'] / (df['operating_hours'] + 1)  # Current wear rate
        
        # Rolling averages for stability
        for col in ['advance_speed', 'total_force', 'earth_pressure']:
            df[f'{col}_ma5'] = df.groupby('cutter_id')[col].rolling(window=5, min_periods=1).mean().reset_index(drop=True)
        
        # 7. WEAR TREND FEATURES
        df['wear_trend'] = df.groupby('cutter_id')['total_wear'].diff().fillna(0)
        df['wear_acceleration_trend'] = df.groupby('cutter_id')['wear_acceleration'].diff().fillna(0)
        
        print(f"Generated {len([col for col in df.columns if col not in ['reading_id', 'cutter_id']])} features for wear prediction")
        
        return df
    
    def prepare_ml_datasets(self, df: pd.DataFrame) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """Prepare datasets for different ML models"""
        
        # Feature sets for different models
        wear_prediction_features = [
            # Geological features
            'ucs_strength', 'abrasivity_index', 'hardness_shore', 'quartz_content',
            'geological_risk_score', 'geological_severity',
            
            # Operational features
            'advance_speed', 'revolution_rpm', 'working_pressure', 'earth_pressure', 'total_force',
            'cutting_intensity', 'pressure_stress', 'force_per_revolution', 'specific_force',
            
            # Cumulative features
            'operating_hours', 'cumulative_cutting_distance', 'cumulative_revolutions',
            'cumulative_force_exposure', 'position_wear_factor',
            
            # Interaction features
            'geology_force_interaction', 'speed_hardness_interaction', 'abrasive_cutting_interaction',
            
            # Temporal features
            'operating_days', 'wear_acceleration',
            
            # Moving averages
            'advance_speed_ma5', 'total_force_ma5', 'earth_pressure_ma5'
        ]
        
        # Encode categorical variables
        df_encoded = df.copy()
        
        # Encode geological severity
        severity_encoder = LabelEncoder()
        df_encoded['geological_severity_encoded'] = severity_encoder.fit_transform(df_encoded['geological_severity'])
        wear_prediction_features.append('geological_severity_encoded')
        
        # Encode geological type
        geology_encoder = LabelEncoder()
        df_encoded['geological_type_encoded'] = geology_encoder.fit_transform(df_encoded['geological_type'])
        wear_prediction_features.append('geological_type_encoded')
        
        # Encode cutter position
        position_encoder = LabelEncoder()
        df_encoded['cutter_position_encoded'] = position_encoder.fit_transform(df_encoded['cutter_position'])
        wear_prediction_features.append('cutter_position_encoded')
        
        # Remove rows with missing values
        available_features = [f for f in wear_prediction_features if f in df_encoded.columns]
        clean_df = df_encoded[available_features + ['next_inspection_wear', 'replacement_needed_in_hours']].dropna()
        
        # Prepare datasets
        datasets = {}
        
        # Dataset 1: Wear progression prediction
        X_wear = clean_df[available_features].values
        y_wear = clean_df['next_inspection_wear'].values
        datasets['wear_progression'] = (X_wear, y_wear)
        
        # Dataset 2: Remaining life prediction
        X_life = clean_df[available_features].values
        y_life = clean_df['replacement_needed_in_hours'].values
        datasets['remaining_life'] = (X_life, y_life)
        
        # Store feature names
        self.wear_features = available_features
        
        print(f"Prepared datasets with {len(available_features)} features")
        print(f"Wear progression dataset: {X_wear.shape[0]} samples")
        print(f"Remaining life dataset: {X_life.shape[0]} samples")
        
        return datasets
    
    def train_wear_prediction_model(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train cutter wear prediction model"""
        
        # Use time series split for temporal data
        tscv = TimeSeriesSplit(n_splits=5)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train ensemble model
        models = {
            'random_forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            'gradient_boosting': GradientBoostingRegressor(n_estimators=100, max_depth=8, random_state=42),
            'ridge': Ridge(alpha=1.0)
        }
        
        model_results = {}
        
        for name, model in models.items():
            # Train model
            model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)
            
            results = {
                'train_r2': r2_score(y_train, y_pred_train),
                'test_r2': r2_score(y_test, y_pred_test),
                'train_mae': mean_absolute_error(y_train, y_pred_train),
                'test_mae': mean_absolute_error(y_test, y_pred_test),
                'train_rmse': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_rmse': np.sqrt(mean_squared_error(y_test, y_pred_test))
            }
            
            model_results[name] = results
            print(f"{name.upper()} - R²: {results['test_r2']:.3f}, MAE: {results['test_mae']:.3f}")
        
        # Select best model based on test R²
        best_model_name = max(model_results.keys(), key=lambda k: model_results[k]['test_r2'])
        self.wear_prediction_model = models[best_model_name]
        
        print(f"Selected {best_model_name} as primary wear prediction model")
        
        # Feature importance analysis
        if hasattr(self.wear_prediction_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.wear_features,
                'importance': self.wear_prediction_model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            print("\nTop 10 Most Important Features:")
            for i, row in feature_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.3f}")
        
        self.model_performance['wear_prediction'] = model_results[best_model_name]
        return model_results
    
    def train_geological_correlation_model(self, df: pd.DataFrame) -> Dict:
        """Train model to correlate geological conditions with optimal parameters"""
        
        print("Training geological correlation model...")
        
        # Create optimal parameter targets based on wear performance
        df_analysis = df.copy()
        
        # Define optimal parameters for each geological type based on minimal wear
        geological_optima = df_analysis.groupby('geological_type').agg({
            'advance_speed': lambda x: x[df_analysis.loc[x.index, 'wear_rate'] == df_analysis.loc[x.index, 'wear_rate'].min()].iloc[0],
            'revolution_rpm': lambda x: x[df_analysis.loc[x.index, 'wear_rate'] == df_analysis.loc[x.index, 'wear_rate'].min()].iloc[0],
            'working_pressure': lambda x: x[df_analysis.loc[x.index, 'wear_rate'] == df_analysis.loc[x.index, 'wear_rate'].min()].iloc[0],
            'wear_rate': 'mean'
        }).reset_index()
        
        # Features for geological correlation
        geological_features = [
            'ucs_strength', 'abrasivity_index', 'hardness_shore', 'quartz_content',
            'geological_risk_score'
        ]
        
        # Prepare training data
        X_geo = df_analysis[geological_features].drop_duplicates()
        
        # Create targets (optimal parameters)
        geo_mapping = df_analysis.groupby(geological_features)[['advance_speed', 'revolution_rpm', 'working_pressure']].mean()
        
        # Train separate models for each parameter
        parameter_models = {}
        
        for param in ['advance_speed', 'revolution_rpm', 'working_pressure']:
            y_param = geo_mapping[param].values
            X_param = geo_mapping.index.to_frame().values
            
            # Train model
            model = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
            model.fit(X_param, y_param)
            
            parameter_models[param] = model
            
            # Evaluate
            y_pred = model.predict(X_param)
            r2 = r2_score(y_param, y_pred)
            print(f"Geological correlation for {param}: R² = {r2:.3f}")
        
        self.parameter_optimizer = parameter_models
        
        results = {
            'geological_optima': geological_optima,
            'correlation_strength': {param: r2_score(geo_mapping[param], model.predict(geo_mapping.index.to_frame().values))
                                   for param, model in parameter_models.items()}
        }
        
        self.model_performance['geological_correlation'] = results
        return results
    
    def predict_cutter_wear(self, current_conditions: Dict) -> Dict:
        """Predict cutter wear for given conditions"""
        
        if self.wear_prediction_model is None:
            return {'error': 'Wear prediction model not trained'}
        
        # Prepare feature vector
        feature_vector = np.array([current_conditions.get(feature, 0) for feature in self.wear_features]).reshape(1, -1)
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Make prediction
        predicted_wear = self.wear_prediction_model.predict(feature_vector_scaled)[0]
        
        # Classify wear condition
        wear_condition = self.classify_wear_condition(predicted_wear)
        
        # Calculate remaining life
        current_wear = current_conditions.get('total_wear', 0)
        wear_rate = (predicted_wear - current_wear) / 24  # Assuming 24-hour prediction
        remaining_life = (self.wear_thresholds['replacement_needed'] - current_wear) / (wear_rate + 0.001)
        
        return {
            'predicted_wear_24h': predicted_wear,
            'current_wear': current_wear,
            'wear_rate_per_hour': wear_rate,
            'wear_condition': wear_condition,
            'remaining_life_hours': max(0, remaining_life),
            'replacement_recommended': wear_condition in ['heavy_wear', 'replacement_needed']
        }
    
    def optimize_boring_parameters(self, geological_conditions: Dict) -> Dict:
        """Optimize boring parameters for given geological conditions"""
        
        if self.parameter_optimizer is None:
            return {'error': 'Parameter optimization model not trained'}
        
        # Prepare geological feature vector
        geo_features = ['ucs_strength', 'abrasivity_index', 'hardness_shore', 'quartz_content', 'geological_risk_score']
        geo_vector = np.array([geological_conditions.get(feature, 0) for feature in geo_features]).reshape(1, -1)
        
        # Predict optimal parameters
        optimal_params = {}
        for param, model in self.parameter_optimizer.items():
            optimal_params[param] = model.predict(geo_vector)[0]
        
        # Add safety margins and practical constraints
        optimal_params['advance_speed'] = np.clip(optimal_params['advance_speed'], 10, 80)
        optimal_params['revolution_rpm'] = np.clip(optimal_params['revolution_rpm'], 5, 12)
        optimal_params['working_pressure'] = np.clip(optimal_params['working_pressure'], 120, 250)
        
        # Calculate expected wear rate with optimal parameters
        expected_wear_rate = self.calculate_wear_rate(
            geological_conditions.get('geological_type', 'mixed_ground'),
            optimal_params['advance_speed'],
            optimal_params['revolution_rpm'],
            geological_conditions.get('earth_pressure', 150),
            geological_conditions.get('total_force', 800),
            1.0  # Per hour calculation
        )
        
        return {
            'optimal_advance_speed': optimal_params['advance_speed'],
            'optimal_revolution_rpm': optimal_params['revolution_rpm'],
            'optimal_working_pressure': optimal_params['working_pressure'],
            'expected_wear_rate': expected_wear_rate,
            'geological_adaptations': self.get_geological_recommendations(geological_conditions)
        }
    
    def get_geological_recommendations(self, geo_conditions: Dict) -> List[str]:
        """Get specific recommendations based on geological conditions"""
        
        recommendations = []
        
        geology_type = geo_conditions.get('geological_type', 'mixed_ground')
        ucs = geo_conditions.get('ucs_strength', 0)
        abrasivity = geo_conditions.get('abrasivity_index', 0)
        
        if geology_type in ['hard_rock', 'medium_rock']:
            recommendations.append("Use carbide-tipped cutters for hard rock conditions")
            recommendations.append("Reduce advance speed to minimize cutter wear")
            recommendations.append("Increase cutting pressure for better penetration")
        
        elif geology_type in ['gravel', 'dense_sand']:
            recommendations.append("Monitor cutter wear closely in abrasive conditions")
            recommendations.append("Consider disc cutter configuration optimization")
            recommendations.append("Maintain consistent advance speed")
        
        elif geology_type in ['soft_clay', 'loose_sand']:
            recommendations.append("Optimize advance speed for maximum productivity")
            recommendations.append("Monitor for clogging in sticky conditions")
            recommendations.append("Adjust working pressure for soil conditions")
        
        if abrasivity > 5.0:
            recommendations.append("High abrasivity detected - implement frequent inspections")
            recommendations.append("Consider wear-resistant cutter materials")
        
        if ucs > 100:
            recommendations.append("Very high rock strength - reduce RPM to prevent shock loading")
            recommendations.append("Implement vibration monitoring")
        
        return recommendations
    
    def generate_maintenance_schedule(self, cutters_data: List[Dict]) -> Dict:
        """Generate optimal maintenance schedule based on wear predictions"""
        
        maintenance_schedule = {
            'immediate_action': [],
            'scheduled_maintenance': [],
            'inspection_due': [],
            'replacement_schedule': []
        }
        
        for cutter_data in cutters_data:
            cutter_id = cutter_data.get('cutter_id', 'Unknown')
            wear_prediction = self.predict_cutter_wear(cutter_data)
            
            if wear_prediction.get('replacement_recommended', False):
                maintenance_schedule['immediate_action'].append({
                    'cutter_id': cutter_id,
                    'action': 'Replace cutter',
                    'priority': 'High',
                    'current_wear': wear_prediction.get('current_wear', 0),
                    'condition': wear_prediction.get('wear_condition', 'Unknown')
                })
            
            elif wear_prediction.get('remaining_life_hours', 1000) < 100:
                maintenance_schedule['scheduled_maintenance'].append({
                    'cutter_id': cutter_id,
                    'action': 'Schedule replacement',
                    'estimated_hours': wear_prediction.get('remaining_life_hours', 0),
                    'condition': wear_prediction.get('wear_condition', 'Unknown')
                })
            
            elif wear_prediction.get('remaining_life_hours', 1000) < 200:
                maintenance_schedule['inspection_due'].append({
                    'cutter_id': cutter_id,
                    'action': 'Detailed inspection',
                    'estimated_hours': wear_prediction.get('remaining_life_hours', 0),
                    'condition': wear_prediction.get('wear_condition', 'Unknown')
                })
        
        # Generate replacement schedule
        replacement_items = maintenance_schedule['scheduled_maintenance']
        replacement_items.sort(key=lambda x: x['estimated_hours'])
        
        for i, item in enumerate(replacement_items):
            maintenance_schedule['replacement_schedule'].append({
                'order': i + 1,
                'cutter_id': item['cutter_id'],
                'estimated_date': f"In {item['estimated_hours']:.0f} hours",
                'priority_level': 'High' if item['estimated_hours'] < 48 else 'Medium'
            })
        
        return maintenance_schedule
    
    def create_wear_analysis_report(self, df: pd.DataFrame) -> str:
        """Generate comprehensive wear analysis report"""
        
        report = []
        report.append("="*80)
        report.append("MTBM CUTTER WEAR ANALYSIS REPORT")
        report.append("="*80)
        
        # Overall statistics
        report.append(f"\nOVERALL WEAR STATISTICS:")
        report.append(f"Total cutters analyzed: {df['cutter_id'].nunique()}")
        report.append(f"Operating period: {df['operating_hours'].max():.1f} hours")
        report.append(f"Tunnel distance: {df['tunnel_distance'].max():.1f}m")
        
        # Wear distribution
        wear_counts = df.groupby('wear_condition').size()
        total_cutters = len(df)
        
        report.append(f"\nWEAR CONDITION DISTRIBUTION:")
        for condition, count in wear_counts.items():
            percentage = (count / total_cutters) * 100
            report.append(f"  {condition}: {count} cutters ({percentage:.1f}%)")
        
        # Geological impact analysis
        geo_wear = df.groupby('geological_type')['wear_rate'].agg(['mean', 'std', 'count'])
        
        report.append(f"\nWEAR RATE BY GEOLOGICAL CONDITIONS:")
        for geology, stats in geo_wear.iterrows():
            report.append(f"  {geology}: {stats['mean']:.3f} ± {stats['std']:.3f} mm/hr ({stats['count']} samples)")
        
        # Position-based analysis
        position_wear = df.groupby('cutter_position')['total_wear'].agg(['mean', 'max'])
        
        report.append(f"\nWEAR BY CUTTER POSITION:")
        for position, stats in position_wear.iterrows():
            report.append(f"  {position}: Mean {stats['mean']:.2f}mm, Max {stats['max']:.2f}mm")
        
        # Recommendations
        report.append(f"\nOPERATIONAL RECOMMENDATIONS:")
        
        high_wear_geology = geo_wear['mean'].idxmax()
        report.append(f"• Highest wear in {high_wear_geology} - implement enhanced monitoring")
        
        critical_cutters = df[df['wear_condition'].isin(['heavy_wear', 'replacement_needed'])]
        if not critical_cutters.empty:
            report.append(f"• {len(critical_cutters)} cutters require immediate attention")
        
        avg_wear_rate = df['wear_rate'].mean()
        if avg_wear_rate > 0.2:
            report.append(f"• High average wear rate ({avg_wear_rate:.3f} mm/hr) - review operating parameters")
        
        report.append("\n" + "="*80)
        
        return "\n".join(report)


def main():
    """Demonstrate the cutter wear prediction ML framework"""
    
    print("MTBM CUTTER WEAR PREDICTION & GEOLOGICAL CORRELATION ML FRAMEWORK")
    print("="*80)
    
    # Initialize framework
    cutter_ml = CutterWearPredictionML()
    
    # Load and process data
    print("Loading cutter wear data...")
    df = cutter_ml.load_cutter_wear_data()
    
    print("Engineering wear prediction features...")
    df = cutter_ml.engineer_wear_features(df)
    
    print(f"Dataset: {df.shape[0]} readings, {df.shape[1]} features")
    
    # Prepare ML datasets
    datasets = cutter_ml.prepare_ml_datasets(df)
    
    # Train wear prediction model
    print("\nTraining cutter wear prediction model...")
    if 'wear_progression' in datasets:
        X_wear, y_wear = datasets['wear_progression']
        wear_results = cutter_ml.train_wear_prediction_model(X_wear, y_wear)
    
    # Train geological correlation model
    print("\nTraining geological correlation model...")
    geo_results = cutter_ml.train_geological_correlation_model(df)
    
    # Demonstrate predictions
    print("\n" + "="*60)
    print("PREDICTION DEMONSTRATIONS")
    print("="*60)
    
    # Example 1: Wear prediction
    sample_conditions = {
        'ucs_strength': 50.0,
        'abrasivity_index': 5.5,
        'hardness_shore': 60,
        'quartz_content': 30,
        'geological_risk_score': 45,
        'geological_severity_encoded': 2,
        'advance_speed': 45,
        'revolution_rpm': 8.5,
        'working_pressure': 180,
        'earth_pressure': 150,
        'total_force': 900,
        'operating_hours': 100,
        'total_wear': 3.5,
        'cutting_intensity': 3.825,
        'pressure_stress': 0.833,
        'cumulative_cutting_distance': 2.5,
        'position_wear_factor': 1.2,
        'geological_type_encoded': 7,
        'cutter_position_encoded': 1
    }
    
    # Fill in missing features with reasonable defaults
    for feature in cutter_ml.wear_features:
        if feature not in sample_conditions:
            sample_conditions[feature] = 0
    
    wear_prediction = cutter_ml.predict_cutter_wear(sample_conditions)
    
    print("CUTTER WEAR PREDICTION EXAMPLE:")
    print(f"  Current wear: {wear_prediction.get('current_wear', 0):.2f}mm")
    print(f"  Predicted 24h wear: {wear_prediction.get('predicted_wear_24h', 0):.2f}mm")
    print(f"  Wear rate: {wear_prediction.get('wear_rate_per_hour', 0):.4f}mm/hr")
    print(f"  Condition: {wear_prediction.get('wear_condition', 'Unknown')}")
    print(f"  Remaining life: {wear_prediction.get('remaining_life_hours', 0):.0f} hours")
    print(f"  Replacement needed: {'Yes' if wear_prediction.get('replacement_recommended', False) else 'No'}")
    
    # Example 2: Parameter optimization
    geological_conditions = {
        'geological_type': 'medium_rock',
        'ucs_strength': 50.0,
        'abrasivity_index': 5.5,
        'hardness_shore': 60,
        'quartz_content': 30,
        'geological_risk_score': 45,
        'earth_pressure': 150,
        'total_force': 900
    }
    
    parameter_optimization = cutter_ml.optimize_boring_parameters(geological_conditions)
    
    print("\nBORING PARAMETER OPTIMIZATION:")
    print(f"  Optimal advance speed: {parameter_optimization.get('optimal_advance_speed', 0):.1f} mm/min")
    print(f"  Optimal RPM: {parameter_optimization.get('optimal_revolution_rpm', 0):.1f}")
    print(f"  Optimal pressure: {parameter_optimization.get('optimal_working_pressure', 0):.0f} bar")
    print(f"  Expected wear rate: {parameter_optimization.get('expected_wear_rate', 0):.4f} mm/hr")
    
    print("\n  Geological Recommendations:")
    for rec in parameter_optimization.get('geological_adaptations', []):
        print(f"    • {rec}")
    
    # Generate maintenance schedule
    sample_cutters = [sample_conditions.copy() for i in range(5)]
    for i, cutter in enumerate(sample_cutters):
        cutter['cutter_id'] = i + 1
        cutter['total_wear'] = 2.0 + i * 2.5  # Varying wear levels
    
    maintenance_schedule = cutter_ml.generate_maintenance_schedule(sample_cutters)
    
    print("\nMAINTENANCE SCHEDULE:")
    for category, items in maintenance_schedule.items():
        if items:
            print(f"  {category.upper()}:")
            for item in items:
                print(f"    • Cutter {item.get('cutter_id', 'N/A')}: {item.get('action', 'N/A')}")
    
    # Generate comprehensive report
    print("\n" + "="*80)
    report = cutter_ml.create_wear_analysis_report(df)
    print(report)
    
    return cutter_ml, df


if __name__ == "__main__":
    framework, data = main()