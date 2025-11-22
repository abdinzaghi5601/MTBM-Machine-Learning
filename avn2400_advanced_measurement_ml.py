#!/usr/bin/env python3
"""
AVN 2400 Advanced Measurement Protocol ML Framework
==================================================

Specialized machine learning system for AVN 2400 advanced measurement protocol focusing on:
1. High-precision measurement accuracy prediction
2. Quality control and compliance monitoring
3. Environmental factor impact analysis
4. Advanced statistical process control
5. Measurement uncertainty quantification

Key Features:
- Precision measurement analytics (sub-millimeter accuracy)
- Quality index prediction and optimization
- Environmental compensation algorithms
- Statistical process control (SPC) implementation
- Measurement system analysis (MSA)
- Compliance tracking and reporting

Author: ML for Tunneling Project
Date: November 2024
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.linear_model import Ridge, LinearRegression, BayesianRidge
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings
warnings.filterwarnings('ignore')

class AVN2400AdvancedMeasurementML:
    """
    Advanced Machine Learning framework for AVN 2400 measurement protocol
    
    Focuses on high-precision measurement analytics, quality control,
    and environmental factor compensation for optimal measurement accuracy.
    """
    
    def __init__(self):
        # Model components
        self.accuracy_model = None
        self.quality_model = None
        self.environmental_model = None
        self.uncertainty_model = None
        self.anomaly_detector = None
        
        # Data processing
        self.scaler = StandardScaler()
        self.robust_scaler = RobustScaler()
        self.poly_features = PolynomialFeatures(degree=2, include_bias=False)
        
        # Feature tracking
        self.feature_columns = []
        self.measurement_targets = []
        self.quality_targets = []
        
        # Performance metrics
        self.model_performance = {}
        self.measurement_statistics = {}
        
        # Quality control parameters
        self.control_limits = {}
        self.specification_limits = {
            'measurement_accuracy': {'lower': 0.0, 'upper': 2.0},  # mm
            'quality_index': {'lower': 70, 'upper': 100},  # percentage
            'environmental_temp': {'lower': 10, 'upper': 40},  # Celsius
            'precision_score': {'lower': 0.8, 'upper': 1.0}  # normalized
        }
        
        # Environmental compensation factors
        self.environmental_factors = {
            'temperature': {
                'optimal_range': (20, 25),  # Celsius
                'compensation_factor': 0.02,  # mm per degree
                'impact_threshold': 5.0  # degrees from optimal
            },
            'humidity': {
                'optimal_range': (45, 65),  # percentage
                'compensation_factor': 0.01,  # mm per percent
                'impact_threshold': 15.0  # percent from optimal
            },
            'pressure': {
                'optimal_range': (1010, 1020),  # hPa
                'compensation_factor': 0.005,  # mm per hPa
                'impact_threshold': 20.0  # hPa from optimal
            }
        }
    
    def generate_synthetic_avn2400_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic AVN 2400 measurement data for demonstration
        """
        print("Generating synthetic AVN 2400 measurement data...")
        
        np.random.seed(42)
        
        # Base measurement conditions
        base_data = {
            'timestamp': pd.date_range('2024-01-01', periods=n_samples, freq='2H'),
            'measurement_id': [f"M{i:04d}" for i in range(n_samples)],
            'tunnel_length_m': np.linspace(0, 500, n_samples)
        }
        
        # Environmental conditions (with daily and seasonal patterns)
        hours = np.array([ts.hour for ts in base_data['timestamp']])
        days = np.array([ts.dayofyear for ts in base_data['timestamp']])
        
        # Temperature with daily and seasonal cycles
        daily_temp_cycle = 5 * np.sin(2 * np.pi * hours / 24)  # Daily variation
        seasonal_temp_cycle = 10 * np.sin(2 * np.pi * days / 365)  # Seasonal variation
        base_temp = 22 + daily_temp_cycle + seasonal_temp_cycle + np.random.normal(0, 2, n_samples)
        
        # Humidity (inversely related to temperature)
        base_humidity = 60 - 0.5 * (base_temp - 22) + np.random.normal(0, 8, n_samples)
        base_humidity = np.clip(base_humidity, 30, 90)
        
        # Atmospheric pressure (with weather patterns)
        pressure_trend = np.cumsum(np.random.normal(0, 0.5, n_samples))
        base_pressure = 1013 + pressure_trend + np.random.normal(0, 3, n_samples)
        
        # Environmental data
        environmental_data = {
            'environmental_temp_c': base_temp,
            'humidity_percent': base_humidity,
            'atmospheric_pressure_hpa': base_pressure
        }
        
        # Calculate environmental impact on measurement accuracy
        measurement_accuracy = []
        quality_index = []
        precision_score = []
        compliance_rating = []
        
        for i in range(n_samples):
            temp = environmental_data['environmental_temp_c'][i]
            humidity = environmental_data['humidity_percent'][i]
            pressure = environmental_data['atmospheric_pressure_hpa'][i]
            
            # Base measurement accuracy (ideal conditions)
            base_accuracy = 0.3  # mm
            
            # Environmental impacts
            temp_impact = abs(temp - 22.5) * self.environmental_factors['temperature']['compensation_factor']
            humidity_impact = abs(humidity - 55) * self.environmental_factors['humidity']['compensation_factor']
            pressure_impact = abs(pressure - 1013) * self.environmental_factors['pressure']['compensation_factor']
            
            # Random measurement noise
            random_noise = np.random.exponential(0.2)
            
            # Total measurement accuracy
            total_accuracy = base_accuracy + temp_impact + humidity_impact + pressure_impact + random_noise
            measurement_accuracy.append(total_accuracy)
            
            # Quality index (inversely related to accuracy)
            quality = 100 - (total_accuracy / 2.0) * 30  # Scale to 0-100
            quality = max(50, min(100, quality + np.random.normal(0, 5)))
            quality_index.append(quality)
            
            # Precision score (consistency measure)
            precision = 1.0 - (total_accuracy / 3.0)  # Higher accuracy = higher precision
            precision = max(0.5, min(1.0, precision + np.random.normal(0, 0.05)))
            precision_score.append(precision)
            
            # Compliance rating based on quality
            if quality >= 90:
                compliance = 'Excellent'
            elif quality >= 80:
                compliance = 'Good'
            elif quality >= 70:
                compliance = 'Acceptable'
            else:
                compliance = 'Poor'
            compliance_rating.append(compliance)
        
        # Measurement data
        measurement_data = {
            'measurement_accuracy_mm': measurement_accuracy,
            'quality_index': quality_index,
            'precision_score': precision_score,
            'compliance_rating': compliance_rating
        }
        
        # Operational data
        operational_data = {
            'operator_id': np.random.choice(['OP001', 'OP002', 'OP003', 'OP004'], n_samples),
            'calibration_status': np.random.choice(['Current', 'Due Soon', 'Overdue'], n_samples, p=[0.8, 0.15, 0.05]),
            'equipment_age_days': np.random.randint(1, 365, n_samples),
            'measurement_duration_min': np.random.uniform(5, 15, n_samples)
        }
        
        # Combine all data
        all_data = {**base_data, **environmental_data, **measurement_data, **operational_data}
        
        return pd.DataFrame(all_data)
    
    def engineer_advanced_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create advanced features for measurement accuracy prediction
        """
        df_features = df.copy()
        
        # Environmental stability indicators
        if 'environmental_temp_c' in df.columns:
            df_features['temp_stability'] = df['environmental_temp_c'].rolling(5).std()
            df_features['temp_trend'] = df['environmental_temp_c'].rolling(5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0)
            df_features['temp_deviation_from_optimal'] = abs(df['environmental_temp_c'] - 22.5)
        
        if 'humidity_percent' in df.columns:
            df_features['humidity_stability'] = df['humidity_percent'].rolling(5).std()
            df_features['humidity_deviation_from_optimal'] = abs(df['humidity_percent'] - 55)
        
        if 'atmospheric_pressure_hpa' in df.columns:
            df_features['pressure_stability'] = df['atmospheric_pressure_hpa'].rolling(5).std()
            df_features['pressure_trend'] = df['atmospheric_pressure_hpa'].rolling(5).apply(
                lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 5 else 0)
        
        # Environmental comfort index
        if all(col in df.columns for col in ['environmental_temp_c', 'humidity_percent', 'atmospheric_pressure_hpa']):
            # Normalized environmental conditions (0 = optimal, 1 = extreme)
            temp_norm = df_features['temp_deviation_from_optimal'] / 20
            humidity_norm = df_features['humidity_deviation_from_optimal'] / 30
            pressure_norm = abs(df['atmospheric_pressure_hpa'] - 1013) / 50
            
            df_features['environmental_comfort_index'] = 1 - np.sqrt(temp_norm**2 + humidity_norm**2 + pressure_norm**2) / 3
        
        # Measurement system features
        if 'measurement_accuracy_mm' in df.columns:
            df_features['accuracy_trend'] = df['measurement_accuracy_mm'].rolling(10).mean()
            df_features['accuracy_volatility'] = df['measurement_accuracy_mm'].rolling(10).std()
            df_features['accuracy_percentile'] = df['measurement_accuracy_mm'].rolling(20).rank(pct=True)
        
        if 'quality_index' in df.columns:
            df_features['quality_trend'] = df['quality_index'].rolling(10).mean()
            df_features['quality_stability'] = df['quality_index'].rolling(10).std()
            df_features['quality_improvement'] = df['quality_index'].diff()
        
        # Equipment condition features
        if 'equipment_age_days' in df.columns:
            df_features['equipment_age_normalized'] = df['equipment_age_days'] / 365
            df_features['equipment_condition_score'] = 1 - (df['equipment_age_days'] / 365) * 0.2
        
        if 'calibration_status' in df.columns:
            # Encode calibration status
            calibration_mapping = {'Current': 1.0, 'Due Soon': 0.7, 'Overdue': 0.3}
            df_features['calibration_score'] = df['calibration_status'].map(calibration_mapping)
        
        # Temporal features
        df_features['hour'] = df_features['timestamp'].dt.hour
        df_features['day_of_week'] = df_features['timestamp'].dt.dayofweek
        df_features['month'] = df_features['timestamp'].dt.month
        df_features['is_weekend'] = (df_features['timestamp'].dt.dayofweek >= 5).astype(int)
        df_features['is_business_hours'] = ((df_features['hour'] >= 8) & (df_features['hour'] <= 17)).astype(int)
        
        return df_features
    
    def train_measurement_accuracy_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train models to predict measurement accuracy
        """
        print("Training Measurement Accuracy Prediction Models...")
        
        df_features = self.engineer_advanced_features(df)
        
        # Select features for modeling
        feature_columns = [
            'environmental_temp_c', 'humidity_percent', 'atmospheric_pressure_hpa',
            'temp_stability', 'humidity_stability', 'pressure_stability',
            'environmental_comfort_index', 'equipment_age_normalized',
            'calibration_score', 'hour', 'day_of_week', 'is_weekend'
        ]
        
        # Filter available features
        available_features = [col for col in feature_columns if col in df_features.columns]
        
        if 'measurement_accuracy_mm' not in df_features.columns:
            print("Target variable 'measurement_accuracy_mm' not found")
            return {}
        
        # Prepare data
        X = df_features[available_features].fillna(0)
        y = df_features['measurement_accuracy_mm']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'Bayesian Ridge': BayesianRidge(),
            'Ridge': Ridge(alpha=1.0)
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for linear models, original for tree-based
            if name in ['Bayesian Ridge', 'Ridge']:
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Calculate metrics
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'predictions': y_pred
            }
            
            print(f"  RMSE: {rmse:.4f} mm")
            print(f"  MAE: {mae:.4f} mm")
            print(f"  R²: {r2:.4f}")
            
            # Feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': available_features,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                
                print(f"  Top 5 Important Features:")
                for idx, row in importance_df.head().iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")
        
        # Store best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.accuracy_model = results[best_model_name]['model']
        self.feature_columns = available_features
        
        print(f"\nBest Model: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
        
        return results
    
    def train_quality_control_model(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Train models for quality index prediction and control
        """
        print("\nTraining Quality Control Models...")
        
        df_features = self.engineer_advanced_features(df)
        
        if 'quality_index' not in df_features.columns:
            print("Target variable 'quality_index' not found")
            return {}
        
        # Features for quality prediction
        quality_features = [
            'measurement_accuracy_mm', 'environmental_temp_c', 'humidity_percent',
            'environmental_comfort_index', 'equipment_condition_score',
            'calibration_score', 'precision_score'
        ]
        
        available_features = [col for col in quality_features if col in df_features.columns]
        
        X = df_features[available_features].fillna(0)
        y = df_features['quality_index']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train quality prediction model
        quality_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        quality_model.fit(X_train, y_train)
        
        y_pred = quality_model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Quality Model Performance:")
        print(f"  RMSE: {rmse:.2f}")
        print(f"  MAE: {mae:.2f}")
        print(f"  R²: {r2:.4f}")
        
        # Calculate control limits for SPC
        self._calculate_control_limits(df_features)
        
        self.quality_model = quality_model
        
        return {
            'model': quality_model,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'control_limits': self.control_limits
        }
    
    def _calculate_control_limits(self, df: pd.DataFrame):
        """
        Calculate statistical process control limits
        """
        if 'measurement_accuracy_mm' in df.columns:
            accuracy_data = df['measurement_accuracy_mm'].dropna()
            
            # X-bar and R chart limits
            mean_accuracy = accuracy_data.mean()
            std_accuracy = accuracy_data.std()
            
            self.control_limits['measurement_accuracy'] = {
                'center_line': mean_accuracy,
                'ucl': mean_accuracy + 3 * std_accuracy,  # Upper Control Limit
                'lcl': max(0, mean_accuracy - 3 * std_accuracy),  # Lower Control Limit
                'usl': self.specification_limits['measurement_accuracy']['upper'],  # Upper Spec Limit
                'lsl': self.specification_limits['measurement_accuracy']['lower']   # Lower Spec Limit
            }
        
        if 'quality_index' in df.columns:
            quality_data = df['quality_index'].dropna()
            
            mean_quality = quality_data.mean()
            std_quality = quality_data.std()
            
            self.control_limits['quality_index'] = {
                'center_line': mean_quality,
                'ucl': min(100, mean_quality + 3 * std_quality),
                'lcl': max(0, mean_quality - 3 * std_quality),
                'usl': self.specification_limits['quality_index']['upper'],
                'lsl': self.specification_limits['quality_index']['lower']
            }
    
    def detect_measurement_anomalies(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect anomalies in measurement data using multiple methods
        """
        print("Detecting Measurement Anomalies...")
        
        df_features = self.engineer_advanced_features(df)
        
        # Features for anomaly detection
        anomaly_features = [
            'measurement_accuracy_mm', 'quality_index', 'environmental_temp_c',
            'humidity_percent', 'precision_score'
        ]
        
        available_features = [col for col in anomaly_features if col in df_features.columns]
        X = df_features[available_features].fillna(0)
        
        # Isolation Forest for anomaly detection
        iso_forest = IsolationForest(contamination=0.05, random_state=42)
        anomaly_scores = iso_forest.fit_predict(X)
        
        # Statistical outlier detection (Z-score method)
        z_scores = np.abs(stats.zscore(X, axis=0))
        statistical_outliers = (z_scores > 3).any(axis=1)
        
        # Control chart violations
        control_violations = []
        if 'measurement_accuracy_mm' in df_features.columns and 'measurement_accuracy' in self.control_limits:
            limits = self.control_limits['measurement_accuracy']
            violations = (df_features['measurement_accuracy_mm'] > limits['ucl']) | \
                        (df_features['measurement_accuracy_mm'] < limits['lcl'])
            control_violations = violations.values
        else:
            control_violations = np.zeros(len(df_features), dtype=bool)
        
        # Combine anomaly detection results
        df_features['anomaly_isolation_forest'] = (anomaly_scores == -1)
        df_features['anomaly_statistical'] = statistical_outliers
        df_features['anomaly_control_chart'] = control_violations
        df_features['anomaly_combined'] = (
            df_features['anomaly_isolation_forest'] | 
            df_features['anomaly_statistical'] | 
            df_features['anomaly_control_chart']
        )
        
        # Anomaly severity score
        anomaly_severity = []
        for idx, row in df_features.iterrows():
            severity = 0
            if row.get('anomaly_isolation_forest', False):
                severity += 1
            if row.get('anomaly_statistical', False):
                severity += 1
            if row.get('anomaly_control_chart', False):
                severity += 2  # Control chart violations are more serious
            anomaly_severity.append(severity)
        
        df_features['anomaly_severity'] = anomaly_severity
        
        # Summary
        total_anomalies = df_features['anomaly_combined'].sum()
        anomaly_rate = total_anomalies / len(df_features) * 100
        
        print(f"Anomaly Detection Results:")
        print(f"  Total anomalies detected: {total_anomalies}")
        print(f"  Anomaly rate: {anomaly_rate:.2f}%")
        print(f"  Isolation Forest: {df_features['anomaly_isolation_forest'].sum()}")
        print(f"  Statistical outliers: {df_features['anomaly_statistical'].sum()}")
        print(f"  Control chart violations: {df_features['anomaly_control_chart'].sum()}")
        
        self.anomaly_detector = iso_forest
        
        return df_features
    
    def generate_measurement_report(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Generate comprehensive measurement analysis report
        """
        print("\n" + "="*70)
        print("AVN 2400 ADVANCED MEASUREMENT ANALYSIS REPORT")
        print("="*70)
        
        df_features = self.engineer_advanced_features(df)
        
        # Basic statistics
        print("\n1. MEASUREMENT SYSTEM OVERVIEW")
        print("-" * 40)
        print(f"Total measurements: {len(df_features):,}")
        print(f"Date range: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
        
        if 'measurement_accuracy_mm' in df_features.columns:
            accuracy_stats = df_features['measurement_accuracy_mm'].describe()
            print(f"\nMeasurement Accuracy Statistics (mm):")
            print(f"  Mean: {accuracy_stats['mean']:.4f}")
            print(f"  Std Dev: {accuracy_stats['std']:.4f}")
            print(f"  Min: {accuracy_stats['min']:.4f}")
            print(f"  Max: {accuracy_stats['max']:.4f}")
            print(f"  95th percentile: {df_features['measurement_accuracy_mm'].quantile(0.95):.4f}")
        
        # Quality analysis
        print("\n2. QUALITY PERFORMANCE ANALYSIS")
        print("-" * 40)
        
        if 'quality_index' in df_features.columns:
            quality_stats = df_features['quality_index'].describe()
            print(f"Quality Index Statistics:")
            print(f"  Mean: {quality_stats['mean']:.2f}")
            print(f"  Std Dev: {quality_stats['std']:.2f}")
            print(f"  Min: {quality_stats['min']:.2f}")
            print(f"  Max: {quality_stats['max']:.2f}")
        
        if 'compliance_rating' in df_features.columns:
            compliance_dist = df_features['compliance_rating'].value_counts(normalize=True)
            print(f"\nCompliance Distribution:")
            for rating, percentage in compliance_dist.items():
                print(f"  {rating}: {percentage:.1%}")
        
        # Environmental impact analysis
        print("\n3. ENVIRONMENTAL IMPACT ANALYSIS")
        print("-" * 40)
        
        if 'environmental_comfort_index' in df_features.columns:
            comfort_mean = df_features['environmental_comfort_index'].mean()
            print(f"Environmental Comfort Index: {comfort_mean:.3f}")
            
            if comfort_mean < 0.7:
                print("  [WARNING] Environmental conditions frequently suboptimal")
            elif comfort_mean > 0.9:
                print("  [EXCELLENT] Excellent environmental control")
            else:
                print("  [ACCEPTABLE] Acceptable environmental conditions")
        
        # Process capability analysis
        print("\n4. PROCESS CAPABILITY ANALYSIS")
        print("-" * 40)
        
        if 'measurement_accuracy_mm' in df_features.columns:
            accuracy_data = df_features['measurement_accuracy_mm']
            spec_limits = self.specification_limits['measurement_accuracy']
            
            # Calculate Cp and Cpk
            process_std = accuracy_data.std()
            process_mean = accuracy_data.mean()
            
            cp = (spec_limits['upper'] - spec_limits['lower']) / (6 * process_std)
            cpk_upper = (spec_limits['upper'] - process_mean) / (3 * process_std)
            cpk_lower = (process_mean - spec_limits['lower']) / (3 * process_std)
            cpk = min(cpk_upper, cpk_lower)
            
            print(f"Process Capability Indices:")
            print(f"  Cp (Potential): {cp:.3f}")
            print(f"  Cpk (Actual): {cpk:.3f}")
            
            if cpk >= 1.33:
                print("  [EXCELLENT] Process is highly capable")
            elif cpk >= 1.0:
                print("  [GOOD] Process is capable")
            elif cpk >= 0.67:
                print("  [WARNING] Process needs improvement")
            else:
                print("  [CRITICAL] Process is not capable")
        
        # Anomaly analysis
        print("\n5. ANOMALY DETECTION RESULTS")
        print("-" * 40)
        
        df_anomalies = self.detect_measurement_anomalies(df)
        
        if 'anomaly_combined' in df_anomalies.columns:
            anomaly_count = df_anomalies['anomaly_combined'].sum()
            anomaly_rate = anomaly_count / len(df_anomalies) * 100
            
            print(f"Anomalies detected: {anomaly_count} ({anomaly_rate:.2f}%)")
            
            if anomaly_rate > 10:
                print("  [CRITICAL] High anomaly rate - investigate measurement system")
            elif anomaly_rate > 5:
                print("  [WARNING] Moderate anomaly rate - monitor closely")
            else:
                print("  [GOOD] Low anomaly rate - system performing well")
        
        # Recommendations
        print("\n6. RECOMMENDATIONS")
        print("-" * 40)
        
        recommendations = []
        
        # Accuracy recommendations
        if 'measurement_accuracy_mm' in df_features.columns:
            mean_accuracy = df_features['measurement_accuracy_mm'].mean()
            if mean_accuracy > 1.0:
                recommendations.append("[ACCURACY] Implement calibration improvements - current accuracy below target")
            elif mean_accuracy < 0.5:
                recommendations.append("[ACCURACY] Excellent measurement precision achieved")
        
        # Environmental recommendations
        if 'environmental_comfort_index' in df_features.columns:
            comfort_mean = df_features['environmental_comfort_index'].mean()
            if comfort_mean < 0.7:
                recommendations.append("[ENVIRONMENT] Improve environmental controls for better measurement stability")
        
        # Equipment recommendations
        if 'calibration_score' in df_features.columns:
            avg_calibration = df_features['calibration_score'].mean()
            if avg_calibration < 0.8:
                recommendations.append("[EQUIPMENT] Increase calibration frequency - equipment drift detected")
        
        # Process recommendations
        recommendations.append("[MONITORING] Implement real-time SPC monitoring")
        recommendations.append("[AUTOMATION] Deploy automated anomaly detection system")
        recommendations.append("[IMPROVEMENT] Establish continuous improvement program")
        
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        print("=" * 70)
        
        return {
            'measurement_statistics': df_features['measurement_accuracy_mm'].describe().to_dict() if 'measurement_accuracy_mm' in df_features.columns else {},
            'quality_statistics': df_features['quality_index'].describe().to_dict() if 'quality_index' in df_features.columns else {},
            'process_capability': {'cp': cp, 'cpk': cpk} if 'measurement_accuracy_mm' in df_features.columns else {},
            'anomaly_rate': anomaly_rate if 'anomaly_combined' in df_anomalies.columns else 0,
            'recommendations': recommendations
        }

def main():
    """
    Main execution function for AVN 2400 analysis
    """
    print("AVN 2400 Advanced Measurement ML Framework")
    print("==========================================")
    
    # Initialize framework
    avn2400 = AVN2400AdvancedMeasurementML()
    
    # Generate synthetic data
    df = avn2400.generate_synthetic_avn2400_data(n_samples=1000)
    
    # Train measurement accuracy model
    accuracy_results = avn2400.train_measurement_accuracy_model(df)
    
    # Train quality control model
    quality_results = avn2400.train_quality_control_model(df)
    
    # Generate comprehensive report
    report = avn2400.generate_measurement_report(df)
    
    # Save results
    print("\nSaving AVN 2400 analysis results...")
    df.to_csv('avn2400_measurement_data.csv', index=False)
    
    print("AVN 2400 analysis complete!")

if __name__ == "__main__":
    main()
