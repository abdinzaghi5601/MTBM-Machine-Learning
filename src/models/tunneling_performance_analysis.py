"""
Tunneling Performance Analysis - Main ML Framework

This module provides comprehensive machine learning analysis for tunneling operations,
including performance prediction, deviation analysis, and operational optimization.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')


class TunnelingPerformanceAnalyzer:
    """
    Comprehensive ML framework for tunneling performance analysis and optimization.
    """
    
    def __init__(self):
        """Initialize the analyzer with default configurations."""
        self.models = {}
        self.scalers = {}
        self.encoders = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
        # Define feature groups for analysis
        self.feature_groups = {
            'geological': ['geological_type', 'ucs_strength', 'abrasivity_index'],
            'operational': ['advance_speed', 'revolution_rpm', 'working_pressure', 'total_thrust'],
            'steering': ['steering_cylinder_top', 'steering_cylinder_bottom', 
                        'steering_cylinder_left', 'steering_cylinder_right'],
            'performance': ['specific_energy', 'cutting_efficiency', 'pressure_efficiency'],
            'deviation': ['horizontal_deviation_machine', 'vertical_deviation_machine',
                         'total_deviation_machine', 'alignment_quality']
        }
    
    def load_and_prepare_data(self, data_path: str = "data/synthetic/tunneling_performance_data.csv"):
        """Load and prepare the tunneling performance data."""
        print("Loading tunneling performance data...")
        
        # Load data
        self.df = pd.read_csv(data_path)
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        print(f"Loaded {len(self.df):,} records")
        print(f"Date range: {self.df['timestamp'].min()} to {self.df['timestamp'].max()}")
        print(f"Tunnel length: {self.df['chainage'].max():.1f}m")
        
        # Encode categorical variables
        le = LabelEncoder()
        self.df['geological_type_encoded'] = le.fit_transform(self.df['geological_type'])
        self.encoders['geological_type'] = le
        
        return self.df
    
    def exploratory_data_analysis(self):
        """Perform comprehensive exploratory data analysis."""
        print("\n" + "="*60)
        print("EXPLORATORY DATA ANALYSIS")
        print("="*60)
        
        # Basic statistics
        print("\nDataset Overview:")
        print(f"Shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        print(f"Duplicate records: {self.df.duplicated().sum()}")
        
        # Geological distribution
        print("\nGeological Distribution:")
        geo_dist = self.df['geological_type'].value_counts()
        for geo_type, count in geo_dist.items():
            print(f"  {geo_type}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        # Performance summary by geological type
        print("\nPerformance by Geological Type:")
        performance_by_geo = self.df.groupby('geological_type').agg({
            'advance_speed': ['mean', 'std'],
            'total_deviation_machine': ['mean', 'std'],
            'alignment_quality': ['mean', 'std'],
            'cutter_wear_rate': ['mean', 'std']
        }).round(3)
        print(performance_by_geo)
        
        # Quality distribution
        print("\nQuality Distribution:")
        quality_dist = self.df['deviation_quality'].value_counts()
        for quality, count in quality_dist.items():
            print(f"  {quality}: {count:,} ({count/len(self.df)*100:.1f}%)")
        
        return performance_by_geo
    
    def create_visualizations(self, save_plots: bool = True):
        """Create comprehensive data visualizations."""
        print("\nGenerating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Tunneling Performance Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. Advance Speed by Geological Type
        sns.boxplot(data=self.df, x='geological_type', y='advance_speed', ax=axes[0,0])
        axes[0,0].set_title('Advance Speed by Geological Type')
        axes[0,0].set_xlabel('Geological Type')
        axes[0,0].set_ylabel('Advance Speed (mm/min)')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Deviation vs Chainage
        axes[0,1].scatter(self.df['chainage'], self.df['total_deviation_machine'], 
                         c=self.df['geological_type_encoded'], alpha=0.6, s=20)
        axes[0,1].set_title('Tunnel Deviation Along Alignment')
        axes[0,1].set_xlabel('Chainage (m)')
        axes[0,1].set_ylabel('Total Deviation (mm)')
        
        # 3. Alignment Quality Distribution
        self.df['alignment_quality'].hist(bins=30, ax=axes[0,2], alpha=0.7, edgecolor='black')
        axes[0,2].set_title('Alignment Quality Distribution')
        axes[0,2].set_xlabel('Alignment Quality Score')
        axes[0,2].set_ylabel('Frequency')
        
        # 4. Performance Correlation Heatmap
        performance_cols = ['advance_speed', 'total_deviation_machine', 'alignment_quality',
                           'cutting_efficiency', 'specific_energy', 'cutter_wear_rate']
        corr_matrix = self.df[performance_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
        axes[1,0].set_title('Performance Metrics Correlation')
        
        # 5. Cutter Wear Rate by Ground Type
        sns.violinplot(data=self.df, x='geological_type', y='cutter_wear_rate', ax=axes[1,1])
        axes[1,1].set_title('Cutter Wear Rate Distribution')
        axes[1,1].set_xlabel('Geological Type')
        axes[1,1].set_ylabel('Wear Rate (mm/hr)')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        # 6. Operational Efficiency Trends
        # Group by chainage bins for trend analysis
        self.df['chainage_bin'] = pd.cut(self.df['chainage'], bins=20)
        efficiency_trend = self.df.groupby('chainage_bin')['operational_efficiency'].mean()
        chainage_centers = [interval.mid for interval in efficiency_trend.index]
        axes[1,2].plot(chainage_centers, efficiency_trend.values, marker='o', linewidth=2)
        axes[1,2].set_title('Operational Efficiency Trend')
        axes[1,2].set_xlabel('Chainage (m)')
        axes[1,2].set_ylabel('Operational Efficiency')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            plt.savefig('dashboards/screenshots/performance_overview.png', dpi=300, bbox_inches='tight')
            print("Saved performance overview plot to dashboards/screenshots/")
        
        plt.show()
    
    def prepare_ml_features(self, target_variable: str = 'total_deviation_machine'):
        """Prepare features for machine learning models."""
        print(f"\nPreparing ML features for target: {target_variable}")
        
        # Select features for ML
        feature_columns = [
            # Geological features
            'geological_type_encoded', 'ucs_strength', 'abrasivity_index',
            # Operational parameters
            'advance_speed', 'revolution_rpm', 'working_pressure', 'total_thrust',
            'earth_pressure',
            # Steering features
            'steering_cylinder_range', 'avg_cylinder_stroke', 'cylinder_variance',
            # Performance metrics
            'specific_energy', 'cutting_efficiency', 'pressure_efficiency',
            'ground_resistance', 'penetration_rate', 'pressure_ratio',
            # Temporal features
            'hour', 'day_of_week'
        ]
        
        # Remove target from features if present
        if target_variable in feature_columns:
            feature_columns.remove(target_variable)
        
        # Prepare feature matrix and target
        X = self.df[feature_columns].copy()
        y = self.df[target_variable].copy()
        
        # Handle any remaining missing values
        X = X.fillna(X.mean())
        
        print(f"Feature matrix shape: {X.shape}")
        print(f"Target variable: {target_variable}")
        print(f"Features: {list(X.columns)}")
        
        return X, y, feature_columns
    
    def train_ml_models(self, target_variable: str = 'total_deviation_machine'):
        """Train multiple ML models for performance prediction."""
        print(f"\nTraining ML models for {target_variable} prediction...")
        
        # Prepare data
        X, y, feature_columns = self.prepare_ml_features(target_variable)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=None
        )
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        self.scalers[target_variable] = scaler
        
        # Define models
        models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Use scaled data for linear models, original for tree-based
            if 'Regression' in name:
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
            
            print(f"  RMSE: {rmse:.3f}")
            print(f"  MAE: {mae:.3f}")
            print(f"  R²: {r2:.3f}")
            
            # Store feature importance for tree-based models
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': feature_columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                self.feature_importance[name] = importance_df
        
        self.models[target_variable] = results
        self.performance_metrics[target_variable] = results
        
        return results
    
    def analyze_feature_importance(self, target_variable: str = 'total_deviation_machine'):
        """Analyze and visualize feature importance."""
        print(f"\nAnalyzing feature importance for {target_variable}...")
        
        if target_variable not in self.models:
            print("Models not trained yet. Please run train_ml_models() first.")
            return
        
        # Create feature importance plot
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle(f'Feature Importance Analysis - {target_variable}', fontsize=14, fontweight='bold')
        
        # Random Forest importance
        if 'Random Forest' in self.feature_importance:
            rf_importance = self.feature_importance['Random Forest'].head(10)
            axes[0].barh(rf_importance['feature'], rf_importance['importance'])
            axes[0].set_title('Random Forest Feature Importance')
            axes[0].set_xlabel('Importance Score')
            
            print("\nTop 10 Most Important Features (Random Forest):")
            for idx, row in rf_importance.head(10).iterrows():
                print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Gradient Boosting importance
        if 'Gradient Boosting' in self.feature_importance:
            gb_importance = self.feature_importance['Gradient Boosting'].head(10)
            axes[1].barh(gb_importance['feature'], gb_importance['importance'])
            axes[1].set_title('Gradient Boosting Feature Importance')
            axes[1].set_xlabel('Importance Score')
        
        plt.tight_layout()
        plt.savefig('dashboards/screenshots/ml_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_performance_report(self, target_variable: str = 'total_deviation_machine'):
        """Generate comprehensive performance analysis report."""
        print("\n" + "="*80)
        print("TUNNELING PERFORMANCE ANALYSIS REPORT")
        print("="*80)
        
        if target_variable not in self.models:
            print("Models not trained yet. Please run train_ml_models() first.")
            return
        
        results = self.models[target_variable]
        
        print(f"\nTarget Variable: {target_variable}")
        print(f"Dataset Size: {len(self.df):,} records")
        print(f"Feature Count: {len(self.prepare_ml_features(target_variable)[2])}")
        
        print(f"\nModel Performance Comparison:")
        print("-" * 60)
        print(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        print("-" * 60)
        
        for name, metrics in results.items():
            if name != 'model':
                print(f"{name:<20} {metrics['rmse']:<10.3f} {metrics['mae']:<10.3f} {metrics['r2']:<10.3f}")
        
        # Find best model
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'] if x != 'model' else -1)
        best_metrics = results[best_model_name]
        
        print(f"\nBest Performing Model: {best_model_name}")
        print(f"  R² Score: {best_metrics['r2']:.3f}")
        print(f"  RMSE: {best_metrics['rmse']:.3f}")
        print(f"  MAE: {best_metrics['mae']:.3f}")
        
        # Business insights
        print(f"\nBusiness Insights:")
        print(f"  • Model explains {best_metrics['r2']*100:.1f}% of variance in {target_variable}")
        print(f"  • Average prediction error: ±{best_metrics['mae']:.2f} units")
        print(f"  • Model suitable for: {'Production use' if best_metrics['r2'] > 0.8 else 'Further development'}")
        
        # Recommendations
        print(f"\nRecommendations:")
        if best_metrics['r2'] > 0.8:
            print("  [EXCELLENT] Model performance is excellent - ready for deployment")
        elif best_metrics['r2'] > 0.6:
            print("  [GOOD] Model performance is good - consider feature engineering")
        else:
            print("  [NEEDS IMPROVEMENT] Model performance needs improvement - collect more data")
        
        return best_model_name, best_metrics
    
    def predict_performance(self, input_data: dict, target_variable: str = 'total_deviation_machine'):
        """Make predictions using the best trained model."""
        if target_variable not in self.models:
            print("Models not trained yet. Please run train_ml_models() first.")
            return None
        
        # Find best model
        results = self.models[target_variable]
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'] if x != 'model' else -1)
        best_model = results[best_model_name]['model']
        
        # Prepare input data
        feature_columns = self.prepare_ml_features(target_variable)[2]
        input_df = pd.DataFrame([input_data])
        
        # Make prediction
        if 'Regression' in best_model_name:
            scaler = self.scalers[target_variable]
            input_scaled = scaler.transform(input_df[feature_columns])
            prediction = best_model.predict(input_scaled)[0]
        else:
            prediction = best_model.predict(input_df[feature_columns])[0]
        
        print(f"Prediction using {best_model_name}: {prediction:.3f}")
        return prediction


def main():
    """Main analysis workflow demonstration."""
    print("Tunneling Performance Analytics - ML Framework")
    print("=" * 60)
    
    # Initialize analyzer
    analyzer = TunnelingPerformanceAnalyzer()
    
    # Load and prepare data
    df = analyzer.load_and_prepare_data()
    
    # Perform EDA
    analyzer.exploratory_data_analysis()
    
    # Create visualizations
    analyzer.create_visualizations()
    
    # Train ML models for deviation prediction
    print("\n" + "="*60)
    print("MACHINE LEARNING MODEL TRAINING")
    print("="*60)
    
    analyzer.train_ml_models('total_deviation_machine')
    analyzer.analyze_feature_importance('total_deviation_machine')
    
    # Generate performance report
    analyzer.generate_performance_report('total_deviation_machine')
    
    # Example prediction
    print("\n" + "="*60)
    print("EXAMPLE PREDICTION")
    print("="*60)
    
    sample_input = {
        'geological_type_encoded': 2,  # hard_rock
        'ucs_strength': 2500,
        'abrasivity_index': 0.8,
        'advance_speed': 25,
        'revolution_rpm': 10,
        'working_pressure': 180,
        'total_thrust': 1200,
        'earth_pressure': 160,
        'steering_cylinder_range': 30,
        'avg_cylinder_stroke': 5,
        'cylinder_variance': 100,
        'specific_energy': 48,
        'cutting_efficiency': 2.5,
        'pressure_efficiency': 0.14,
        'ground_resistance': 6.4,
        'penetration_rate': 0.021,
        'pressure_ratio': 0.89,
        'hour': 14,
        'day_of_week': 2
    }
    
    prediction = analyzer.predict_performance(sample_input)
    print(f"Expected deviation for given conditions: {prediction:.2f} mm")


if __name__ == "__main__":
    main()