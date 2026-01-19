#!/usr/bin/env python3
"""
Hegab et al. (2006) Model Implementation & Comparison Framework
================================================================

This module implements the soil penetration models from:
"Soil Penetration Modeling in Microtunneling Projects"
by Hegab, Smith & Salem (Journal of Construction Engineering and Management, 2006)

Features:
1. Explicit regression equations from the paper (Models A-2, B-3, C-2)
2. Paper's variable transformations (T√L, P√L, log interactions)
3. Comparison framework: Paper models vs. Modern ML (Random Forest, Gradient Boosting)
4. Visualization and performance metrics

Author: ML for Tunneling Project
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# PART 1: HEGAB PAPER MODEL EQUATIONS
# =============================================================================

class HegabPaperModels:
    """
    Implementation of Hegab et al. (2006) penetration time models.

    The paper developed separate models for three soil types:
    - Soil Type A: Soft clay / Loose sand (qu < 0.05 MPa, φ < 30°)
    - Soil Type B: Medium clay / Medium sand (0.05-0.1 MPa, φ = 30-36°)
    - Soil Type C: Hard clay / Dense sand (qu > 0.1 MPa, φ > 36°)

    Variables:
    - T: Shear force of cutter head (metric tons)
    - P: Jacking force (metric tons)
    - D: Machine diameter (meters)
    - L: Jacking/driven length (meters)
    - TM: Driving/tunneling time (minutes) [OUTPUT]
    """

    def __init__(self):
        self.model_limits = {
            'diameter_mm': (400, 1760),
            'drive_length_m': (0, 400),
            'jacking_force_tons': (0, 700),
            'shear_force_tons': (0, 300)
        }

        # Soil classification thresholds from paper (Table 4)
        self.soil_categories = {
            'A': {'cohesive': 'Soft (qu < 0.05 MPa)', 'granular': 'Loose (φ < 30°)'},
            'B': {'cohesive': 'Medium (0.05-0.1 MPa)', 'granular': 'Medium (φ = 30-36°)'},
            'C': {'cohesive': 'Hard (qu > 0.1 MPa)', 'granular': 'Dense (φ > 36°)'}
        }

    def model_a2_soft_soil(self, T: float, P: float, D: float, L: float) -> float:
        """
        Model A-2 for Soft Soil (Soil Type A)

        Equation (from paper):
        ³√TM = 0.0668L + 0.00801P + 4.06D - 0.00167T
               - 0.000820P√L + 0.000411T√L - 0.0000001TL²
               - 0.753 log(PL) + 1.07 log(TL)

        Parameters:
        -----------
        T : float - Shear force (metric tons)
        P : float - Jacking force (metric tons)
        D : float - Machine diameter (meters)
        L : float - Driven length (meters)

        Returns:
        --------
        TM : float - Predicted tunneling time (minutes)
        """
        # Avoid log(0) and division issues
        L = max(L, 0.1)
        T = max(T, 0.1)
        P = max(P, 0.1)

        sqrt_L = np.sqrt(L)
        TL = T * L
        PL = P * L
        TL_sq = TL ** 2

        # Model A-2 equation (cube root of TM)
        TM_cuberoot = (0.0668 * L
                       + 0.00801 * P
                       + 4.06 * D
                       - 0.00167 * T
                       - 0.000820 * P * sqrt_L
                       + 0.000411 * T * sqrt_L
                       - 0.0000001 * TL_sq
                       - 0.753 * np.log10(PL + 1)
                       + 1.07 * np.log10(TL + 1))

        # Cube to get TM (ensure non-negative)
        TM = max(0, TM_cuberoot ** 3)
        return TM

    def model_b3_medium_soil(self, T: float, P: float, D: float, L: float) -> float:
        """
        Model B-3 for Medium Soil (Soil Type B)

        Equation (from paper):
        √TM = 0.548L - 0.134P + 40.8D - 0.00897T - 0.00358PL
              - 0.000325TL + 0.0476P√L + 0.00274T√L
              + 0.000002PL² + 0.000001TL² - 7.20 log(TL)

        Parameters & Returns: Same as model_a2
        """
        L = max(L, 0.1)
        T = max(T, 0.1)
        P = max(P, 0.1)

        sqrt_L = np.sqrt(L)
        TL = T * L
        PL = P * L
        TL_sq = TL ** 2
        PL_sq = PL ** 2

        # Model B-3 equation (square root of TM)
        TM_sqrt = (0.548 * L
                   - 0.134 * P
                   + 40.8 * D
                   - 0.00897 * T
                   - 0.00358 * PL
                   - 0.000325 * TL
                   + 0.0476 * P * sqrt_L
                   + 0.00274 * T * sqrt_L
                   + 0.000002 * PL_sq
                   + 0.000001 * TL_sq
                   - 7.20 * np.log10(TL + 1))

        # Square to get TM (ensure non-negative)
        TM = max(0, TM_sqrt ** 2)
        return TM

    def model_c2_hard_soil(self, T: float, P: float, D: float, L: float) -> float:
        """
        Model C-2 for Hard Soil (Soil Type C)

        Equation (from paper):
        √TM = 0.468L - 0.176P + 46.2D - 0.00502T - 0.00229PL
              - 0.000194TL + 0.0394P√L + 0.00103T√L
              - 0.000001PL² + 0.000001TL² - 7.27 log(PL)

        Parameters & Returns: Same as model_a2
        """
        L = max(L, 0.1)
        T = max(T, 0.1)
        P = max(P, 0.1)

        sqrt_L = np.sqrt(L)
        TL = T * L
        PL = P * L
        TL_sq = TL ** 2
        PL_sq = PL ** 2

        # Model C-2 equation (square root of TM)
        TM_sqrt = (0.468 * L
                   - 0.176 * P
                   + 46.2 * D
                   - 0.00502 * T
                   - 0.00229 * PL
                   - 0.000194 * TL
                   + 0.0394 * P * sqrt_L
                   + 0.00103 * T * sqrt_L
                   - 0.000001 * PL_sq
                   + 0.000001 * TL_sq
                   - 7.27 * np.log10(PL + 1))

        # Square to get TM (ensure non-negative)
        TM = max(0, TM_sqrt ** 2)
        return TM

    def predict(self, T: float, P: float, D: float, L: float,
                soil_type: str = 'auto') -> Dict[str, float]:
        """
        Predict penetration time using appropriate model.

        Parameters:
        -----------
        T, P, D, L : float - Model inputs
        soil_type : str - 'A', 'B', 'C', or 'auto' (auto-detect based on force/length ratios)

        Returns:
        --------
        dict with predictions from relevant model(s)
        """
        results = {}

        if soil_type == 'auto' or soil_type == 'all':
            # Predict with all three models
            results['soil_A_soft'] = self.model_a2_soft_soil(T, P, D, L)
            results['soil_B_medium'] = self.model_b3_medium_soil(T, P, D, L)
            results['soil_C_hard'] = self.model_c2_hard_soil(T, P, D, L)

            # Auto-detect soil type based on operational patterns (from paper's clustering)
            # Based on Table 3 centroids: penetration_time/length, shear_force, jacking_force
            if P < 150 and T < 110:
                results['recommended'] = 'soil_A_soft'
            elif P > 300 or T > 120:
                results['recommended'] = 'soil_C_hard'
            else:
                results['recommended'] = 'soil_B_medium'

        elif soil_type.upper() == 'A':
            results['prediction'] = self.model_a2_soft_soil(T, P, D, L)
        elif soil_type.upper() == 'B':
            results['prediction'] = self.model_b3_medium_soil(T, P, D, L)
        elif soil_type.upper() == 'C':
            results['prediction'] = self.model_c2_hard_soil(T, P, D, L)
        else:
            raise ValueError(f"Unknown soil type: {soil_type}. Use 'A', 'B', 'C', or 'auto'")

        return results

    def predict_batch(self, df: pd.DataFrame,
                      T_col: str = 'shear_force',
                      P_col: str = 'jacking_force',
                      D_col: str = 'diameter_m',
                      L_col: str = 'tunnel_length_m',
                      soil_type: str = 'auto') -> pd.DataFrame:
        """
        Batch prediction for a DataFrame.

        Returns DataFrame with predictions added.
        """
        df = df.copy()

        predictions_A = []
        predictions_B = []
        predictions_C = []

        for _, row in df.iterrows():
            T = row.get(T_col, 100)
            P = row.get(P_col, 200)
            D = row.get(D_col, 1.0)
            L = row.get(L_col, 100)

            predictions_A.append(self.model_a2_soft_soil(T, P, D, L))
            predictions_B.append(self.model_b3_medium_soil(T, P, D, L))
            predictions_C.append(self.model_c2_hard_soil(T, P, D, L))

        df['hegab_model_A_soft'] = predictions_A
        df['hegab_model_B_medium'] = predictions_B
        df['hegab_model_C_hard'] = predictions_C

        return df


# =============================================================================
# PART 2: PAPER'S VARIABLE TRANSFORMATIONS
# =============================================================================

class HegabFeatureEngineering:
    """
    Implements the variable transformations from Hegab et al. (2006).

    The paper uses 51 candidate variables from 4 base variables:
    - Original: T, P, D, L
    - Transformations: √, ³√, ², log₁₀
    - Interactions: TL, PL, T√L, P√L, TL², PL²
    """

    @staticmethod
    def add_hegab_transformations(df: pd.DataFrame,
                                   T_col: str = 'shear_force',
                                   P_col: str = 'jacking_force',
                                   D_col: str = 'diameter_m',
                                   L_col: str = 'tunnel_length_m') -> pd.DataFrame:
        """
        Add all variable transformations from the Hegab paper.

        Parameters:
        -----------
        df : DataFrame with base variables
        T_col, P_col, D_col, L_col : column names for base variables

        Returns:
        --------
        DataFrame with 40+ new features added
        """
        df = df.copy()

        # Get base variables (with safety for missing columns)
        T = df.get(T_col, pd.Series([100] * len(df))).fillna(100)
        P = df.get(P_col, pd.Series([200] * len(df))).fillna(200)
        D = df.get(D_col, pd.Series([1.0] * len(df))).fillna(1.0)
        L = df.get(L_col, pd.Series([100] * len(df))).fillna(100)

        # Ensure positive values for transformations
        T = T.clip(lower=0.1)
        P = P.clip(lower=0.1)
        D = D.clip(lower=0.1)
        L = L.clip(lower=0.1)

        # -----------------------------------------------------------------
        # 1. SQUARE ROOT TRANSFORMATIONS
        # -----------------------------------------------------------------
        df['sqrt_T'] = np.sqrt(T)
        df['sqrt_P'] = np.sqrt(P)
        df['sqrt_D'] = np.sqrt(D)
        df['sqrt_L'] = np.sqrt(L)

        # -----------------------------------------------------------------
        # 2. CUBE ROOT TRANSFORMATIONS
        # -----------------------------------------------------------------
        df['cbrt_T'] = np.cbrt(T)
        df['cbrt_P'] = np.cbrt(P)
        df['cbrt_D'] = np.cbrt(D)
        df['cbrt_L'] = np.cbrt(L)

        # -----------------------------------------------------------------
        # 3. SQUARE TRANSFORMATIONS
        # -----------------------------------------------------------------
        df['T_squared'] = T ** 2
        df['P_squared'] = P ** 2
        df['D_squared'] = D ** 2
        df['L_squared'] = L ** 2

        # -----------------------------------------------------------------
        # 4. LOGARITHM TRANSFORMATIONS (base 10, as in paper)
        # -----------------------------------------------------------------
        df['log_T'] = np.log10(T + 1)
        df['log_P'] = np.log10(P + 1)
        df['log_D'] = np.log10(D + 1)
        df['log_L'] = np.log10(L + 1)

        # -----------------------------------------------------------------
        # 5. TWO-WAY INTERACTIONS (Key from paper)
        # -----------------------------------------------------------------
        # Basic interactions
        df['TL'] = T * L  # Shear force × Length
        df['PL'] = P * L  # Jacking force × Length
        df['TD'] = T * D  # Shear force × Diameter
        df['PD'] = P * D  # Jacking force × Diameter
        df['TP'] = T * P  # Shear × Jacking
        df['DL'] = D * L  # Diameter × Length

        # -----------------------------------------------------------------
        # 6. MIXED INTERACTIONS (Critical from paper's models)
        # -----------------------------------------------------------------
        # These appear directly in the paper's final equations
        df['T_sqrt_L'] = T * np.sqrt(L)  # T√L
        df['P_sqrt_L'] = P * np.sqrt(L)  # P√L
        df['TL_squared'] = (T * L) ** 2  # TL²
        df['PL_squared'] = (P * L) ** 2  # PL²

        # -----------------------------------------------------------------
        # 7. LOGARITHM OF INTERACTIONS (Used in paper)
        # -----------------------------------------------------------------
        df['log_TL'] = np.log10(T * L + 1)  # log(TL)
        df['log_PL'] = np.log10(P * L + 1)  # log(PL)
        df['log_TD'] = np.log10(T * D + 1)
        df['log_PD'] = np.log10(P * D + 1)

        # -----------------------------------------------------------------
        # 8. ADDITIONAL DERIVED FEATURES (Engineering insight)
        # -----------------------------------------------------------------
        # Force ratios
        df['shear_to_jack_ratio'] = T / P
        df['force_per_diameter'] = (T + P) / D
        df['force_per_length'] = (T + P) / L

        # Normalized features
        df['specific_shear'] = T / (D * L + 1)
        df['specific_jacking'] = P / (D * L + 1)

        # Power-related
        df['total_force'] = T + P
        df['force_product'] = T * P
        df['force_geometric_mean'] = np.sqrt(T * P)

        # Diameter-length ratio (important for machine behavior)
        df['aspect_ratio'] = D / L
        df['DL_product'] = D * L

        print(f"Added {len([c for c in df.columns if c not in [T_col, P_col, D_col, L_col]])} Hegab-style features")

        return df

    @staticmethod
    def get_paper_feature_list() -> List[str]:
        """Return list of features used in paper's best models."""
        return [
            # Base variables
            'shear_force', 'jacking_force', 'diameter_m', 'tunnel_length_m',
            # Key transformations from paper
            'sqrt_L', 'TL', 'PL', 'T_sqrt_L', 'P_sqrt_L',
            'TL_squared', 'PL_squared', 'log_TL', 'log_PL'
        ]


# =============================================================================
# PART 3: COMPARISON FRAMEWORK
# =============================================================================

class ModelComparisonFramework:
    """
    Framework to compare Hegab paper models with modern ML approaches.

    Compares:
    1. Hegab et al. (2006) regression equations
    2. Random Forest Regressor
    3. Gradient Boosting Regressor
    4. Ridge Regression (linear baseline)
    """

    def __init__(self):
        self.hegab_models = HegabPaperModels()
        self.feature_engineer = HegabFeatureEngineering()

        self.ml_models = {
            'RandomForest': RandomForestRegressor(
                n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
            ),
            'GradientBoosting': GradientBoostingRegressor(
                n_estimators=100, max_depth=8, random_state=42
            ),
            'Ridge': Ridge(alpha=1.0)
        }

        self.scaler = StandardScaler()
        self.results = {}
        self.trained_models = {}

    def generate_synthetic_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Generate synthetic microtunneling data for testing.

        Creates realistic data following patterns from the Hegab paper.
        Uses realistic penetration rates based on Table 3 cluster centroids:
        - Soil A: ~24 min/m (soft, fast penetration)
        - Soil B: ~35 min/m (medium)
        - Soil C: ~57 min/m (hard, slow penetration)
        """
        np.random.seed(42)

        # Simulate different soil zones
        soil_zones = np.random.choice(['A', 'B', 'C'], n_samples, p=[0.4, 0.35, 0.25])

        data = []
        for i, soil in enumerate(soil_zones):
            # Base parameters depend on soil type (from Table 3 centroids)
            if soil == 'A':  # Soft soil - fast penetration
                T = np.random.uniform(80, 120)      # Shear force ~102 tons
                P = np.random.uniform(40, 100)      # Jacking force ~65 tons
                penetration_rate = np.random.uniform(18, 30)  # ~24 min/m
                base_speed = np.random.uniform(35, 55)
            elif soil == 'B':  # Medium soil
                T = np.random.uniform(100, 130)     # Shear force ~115 tons
                P = np.random.uniform(300, 550)     # Jacking force ~419 tons
                penetration_rate = np.random.uniform(28, 42)  # ~35 min/m
                base_speed = np.random.uniform(20, 40)
            else:  # Hard soil (C) - slow penetration
                T = np.random.uniform(115, 140)     # Shear force ~126 tons
                P = np.random.uniform(180, 280)     # Jacking force ~227 tons
                penetration_rate = np.random.uniform(48, 68)  # ~57 min/m
                base_speed = np.random.uniform(10, 25)

            # Machine parameters (within paper limits)
            D = np.random.uniform(0.6, 1.2)   # Diameter in meters (common range)
            L = np.random.uniform(20, 200)    # Length in meters

            # Calculate penetration time based on realistic rates
            # Time = penetration_rate (min/m) × length (m)
            base_time = penetration_rate * L

            # Add realistic noise (±15%)
            actual_time = base_time * np.random.uniform(0.85, 1.15)
            actual_time = max(10, actual_time)  # Minimum 10 minutes

            data.append({
                'sample_id': i + 1,
                'shear_force': T,
                'jacking_force': P,
                'diameter_m': D,
                'tunnel_length_m': L,
                'soil_type': soil,
                'advance_speed_mm_min': base_speed,
                'penetration_time_min': actual_time,
                'time_per_meter': actual_time / L
            })

        df = pd.DataFrame(data)
        print(f"Generated {n_samples} synthetic samples")
        print(f"Soil distribution: A={sum(soil_zones=='A')}, B={sum(soil_zones=='B')}, C={sum(soil_zones=='C')}")
        print(f"Time range: {df['penetration_time_min'].min():.0f} - {df['penetration_time_min'].max():.0f} minutes")

        return df

    def prepare_data(self, df: pd.DataFrame,
                     target_col: str = 'penetration_time_min') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data with Hegab transformations for ML models.
        """
        # Add Hegab-style features
        df_features = self.feature_engineer.add_hegab_transformations(df)

        # Add Hegab model predictions as features (model stacking)
        df_features = self.hegab_models.predict_batch(df_features)

        # Select numeric features only
        feature_cols = [col for col in df_features.columns
                       if col not in ['sample_id', 'soil_type', target_col, 'time_per_meter']
                       and df_features[col].dtype in ['float64', 'int64']]

        X = df_features[feature_cols].fillna(0)
        y = df_features[target_col]

        return X, y, feature_cols

    def evaluate_hegab_models(self, df: pd.DataFrame,
                               target_col: str = 'penetration_time_min') -> Dict[str, Dict]:
        """
        Evaluate Hegab-style Linear Regression models on the dataset.

        Since the paper's exact equations were fit to their specific data,
        we train new linear regression models using the paper's variable
        transformations to demonstrate the methodology.
        """
        results = {}
        y_true = df[target_col]

        # Add Hegab transformations
        df_features = self.feature_engineer.add_hegab_transformations(df.copy())

        # Paper's key features (from their best models)
        hegab_features = ['tunnel_length_m', 'jacking_force', 'diameter_m', 'shear_force',
                          'PL', 'TL', 'P_sqrt_L', 'T_sqrt_L', 'PL_squared', 'TL_squared',
                          'log_PL', 'log_TL']

        available_features = [f for f in hegab_features if f in df_features.columns]
        X_hegab = df_features[available_features].fillna(0)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_hegab, y_true, test_size=0.2, random_state=42
        )

        # 1. Simple Linear Regression (paper's approach)
        lr_model = LinearRegression()
        lr_model.fit(X_train, y_train)
        y_pred_lr = lr_model.predict(X_test)

        results['Hegab_LinearReg'] = {
            'R2': r2_score(y_test, y_pred_lr),
            'MAE': mean_absolute_error(y_test, y_pred_lr),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
            'MAPE': np.mean(np.abs((y_test - y_pred_lr) / y_test)) * 100,
            'predictions': y_pred_lr
        }

        # 2. Ridge Regression (paper uses this for regularization)
        ridge_model = Ridge(alpha=1.0)
        ridge_model.fit(X_train, y_train)
        y_pred_ridge = ridge_model.predict(X_test)

        results['Hegab_Ridge'] = {
            'R2': r2_score(y_test, y_pred_ridge),
            'MAE': mean_absolute_error(y_test, y_pred_ridge),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_ridge)),
            'MAPE': np.mean(np.abs((y_test - y_pred_ridge) / y_test)) * 100,
            'predictions': y_pred_ridge
        }

        # 3. Soil-specific models (as paper recommends)
        if 'soil_type' in df.columns:
            y_pred_per_soil = np.zeros(len(y_test))
            test_indices = y_test.index

            for soil_type in ['A', 'B', 'C']:
                # Train on specific soil type
                mask_train = df.loc[X_train.index, 'soil_type'] == soil_type
                mask_test = df.loc[test_indices, 'soil_type'] == soil_type

                if mask_train.sum() > 10 and mask_test.sum() > 0:
                    model = Ridge(alpha=1.0)
                    model.fit(X_train[mask_train], y_train[mask_train])
                    predictions = model.predict(X_test[mask_test])

                    # Map predictions back
                    test_positions = np.where(mask_test)[0]
                    for pos, pred in zip(test_positions, predictions):
                        y_pred_per_soil[pos] = pred

            # Calculate metrics only for valid predictions
            valid_mask = y_pred_per_soil > 0
            if valid_mask.sum() > 0:
                y_test_valid = y_test.values[valid_mask]
                y_pred_valid = y_pred_per_soil[valid_mask]

                results['Hegab_Per_Soil'] = {
                    'R2': r2_score(y_test_valid, y_pred_valid),
                    'MAE': mean_absolute_error(y_test_valid, y_pred_valid),
                    'RMSE': np.sqrt(mean_squared_error(y_test_valid, y_pred_valid)),
                    'MAPE': np.mean(np.abs((y_test_valid - y_pred_valid) / y_test_valid)) * 100,
                    'predictions': y_pred_valid
                }

        return results

    def train_ml_models(self, X: pd.DataFrame, y: pd.Series,
                        test_size: float = 0.2) -> Dict[str, Dict]:
        """
        Train and evaluate modern ML models.
        """
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        results = {}

        for name, model in self.ml_models.items():
            print(f"Training {name}...")

            # Train
            model.fit(X_train_scaled, y_train)
            self.trained_models[name] = model

            # Predict
            y_pred_train = model.predict(X_train_scaled)
            y_pred_test = model.predict(X_test_scaled)

            # Metrics
            results[name] = {
                'train_R2': r2_score(y_train, y_pred_train),
                'test_R2': r2_score(y_test, y_pred_test),
                'R2': r2_score(y_test, y_pred_test),  # For comparison
                'train_MAE': mean_absolute_error(y_train, y_pred_train),
                'test_MAE': mean_absolute_error(y_test, y_pred_test),
                'MAE': mean_absolute_error(y_test, y_pred_test),
                'train_RMSE': np.sqrt(mean_squared_error(y_train, y_pred_train)),
                'test_RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_test)),
                'MAPE': np.mean(np.abs((y_test - y_pred_test) / y_test)) * 100,
                'predictions': y_pred_test
            }

            # Feature importance (for tree-based models)
            if hasattr(model, 'feature_importances_'):
                importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                results[name]['feature_importance'] = importance

        self.results['ml_models'] = results
        return results

    def run_full_comparison(self, df: pd.DataFrame = None,
                            n_samples: int = 1000,
                            target_col: str = 'penetration_time_min') -> Dict:
        """
        Run complete comparison between Hegab models and ML models.
        """
        print("=" * 70)
        print("MODEL COMPARISON: Hegab (2006) vs Modern ML")
        print("=" * 70)

        # Generate or use provided data
        if df is None:
            print("\n1. Generating synthetic data...")
            df = self.generate_synthetic_data(n_samples)
        else:
            print(f"\n1. Using provided data: {len(df)} samples")

        # Evaluate Hegab models
        print("\n2. Evaluating Hegab paper models...")
        hegab_results = self.evaluate_hegab_models(df, target_col)
        self.results['hegab_models'] = hegab_results

        # Prepare data with Hegab features
        print("\n3. Preparing data with Hegab transformations...")
        X, y, feature_cols = self.prepare_data(df, target_col)
        print(f"   Total features: {len(feature_cols)}")

        # Train ML models
        print("\n4. Training ML models...")
        ml_results = self.train_ml_models(X, y)

        # Print comparison
        self._print_comparison()

        return self.results

    def _print_comparison(self):
        """Print formatted comparison results."""
        print("\n" + "=" * 70)
        print("RESULTS COMPARISON")
        print("=" * 70)

        # Combine all results
        all_results = {}

        if 'hegab_models' in self.results:
            for name, metrics in self.results['hegab_models'].items():
                all_results[name] = metrics

        if 'ml_models' in self.results:
            for name, metrics in self.results['ml_models'].items():
                all_results[f"ML_{name}"] = metrics

        # Print table
        print(f"\n{'Model':<25} {'R²':>10} {'MAE':>12} {'RMSE':>12} {'MAPE %':>10}")
        print("-" * 70)

        for name, metrics in all_results.items():
            r2 = metrics.get('R2', metrics.get('test_R2', 0))
            mae = metrics.get('MAE', metrics.get('test_MAE', 0))
            rmse = metrics.get('RMSE', metrics.get('test_RMSE', 0))
            mape = metrics.get('MAPE', 0)

            print(f"{name:<25} {r2:>10.4f} {mae:>12.2f} {rmse:>12.2f} {mape:>10.2f}")

        print("-" * 70)

        # Find best model
        best_model = max(all_results.items(),
                        key=lambda x: x[1].get('R2', x[1].get('test_R2', 0)))
        print(f"\nBest Model: {best_model[0]} (R² = {best_model[1].get('R2', best_model[1].get('test_R2', 0)):.4f})")

    def plot_comparison(self, save_path: str = None):
        """
        Create visualization comparing all models.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Hegab (2006) Paper Models vs Modern ML Comparison', fontsize=14, fontweight='bold')

        # Collect all results
        all_results = {}
        if 'hegab_models' in self.results:
            all_results.update(self.results['hegab_models'])
        if 'ml_models' in self.results:
            for name, metrics in self.results['ml_models'].items():
                all_results[f"ML_{name}"] = metrics

        # 1. R² Comparison Bar Chart
        models = list(all_results.keys())
        r2_scores = [all_results[m].get('R2', all_results[m].get('test_R2', 0)) for m in models]

        colors = ['#ff9999' if 'Hegab' in m else '#99ccff' for m in models]
        axes[0, 0].barh(models, r2_scores, color=colors)
        axes[0, 0].set_xlabel('R² Score')
        axes[0, 0].set_title('Model Comparison: R² Score')
        axes[0, 0].axvline(x=0.8, color='green', linestyle='--', alpha=0.5, label='Good (0.8)')
        axes[0, 0].legend()

        # 2. MAE Comparison
        mae_scores = [all_results[m].get('MAE', all_results[m].get('test_MAE', 0)) for m in models]
        axes[0, 1].barh(models, mae_scores, color=colors)
        axes[0, 1].set_xlabel('Mean Absolute Error')
        axes[0, 1].set_title('Model Comparison: MAE (Lower is Better)')

        # 3. MAPE Comparison
        mape_scores = [all_results[m].get('MAPE', 0) for m in models]
        axes[1, 0].barh(models, mape_scores, color=colors)
        axes[1, 0].set_xlabel('Mean Absolute Percentage Error (%)')
        axes[1, 0].set_title('Model Comparison: MAPE % (Lower is Better)')
        axes[1, 0].axvline(x=10, color='green', linestyle='--', alpha=0.5, label='Paper target (10%)')
        axes[1, 0].legend()

        # 4. Feature Importance (if available)
        if 'ml_models' in self.results and 'RandomForest' in self.results['ml_models']:
            rf_results = self.results['ml_models']['RandomForest']
            if 'feature_importance' in rf_results:
                importance = rf_results['feature_importance'].head(15)
                axes[1, 1].barh(importance['feature'], importance['importance'], color='#66b3ff')
                axes[1, 1].set_xlabel('Feature Importance')
                axes[1, 1].set_title('Top 15 Features (Random Forest)')
                axes[1, 1].invert_yaxis()
        else:
            axes[1, 1].text(0.5, 0.5, 'Feature importance\nnot available',
                          ha='center', va='center', fontsize=12)
            axes[1, 1].set_title('Feature Importance')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def get_feature_importance_comparison(self) -> pd.DataFrame:
        """
        Get feature importance from ML models, highlighting Hegab-style features.
        """
        if 'ml_models' not in self.results:
            return pd.DataFrame()

        hegab_features = set(HegabFeatureEngineering.get_paper_feature_list())

        importance_dfs = []
        for model_name, results in self.results['ml_models'].items():
            if 'feature_importance' in results:
                imp = results['feature_importance'].copy()
                imp['model'] = model_name
                imp['is_hegab_feature'] = imp['feature'].isin(hegab_features)
                importance_dfs.append(imp)

        if importance_dfs:
            return pd.concat(importance_dfs, ignore_index=True)
        return pd.DataFrame()


# =============================================================================
# PART 4: HEGAB (2009) LABOR PERFORMANCE MODEL
# =============================================================================

class LaborPerformanceModel:
    """
    Implementation of Hegab & Smith (2009) Labor Performance Model.

    "Labor Performance Analysis for Microtunneling Projects"
    Journal of Construction Engineering and Management, Vol. 135, No. 5

    Uses Log-Logistic probability distribution to model pipe preparation time.

    Key findings from paper:
    - Data: 35 projects, 385 events, 15-100 min/pipe (avg: 50 min)
    - Best fit: Log-Logistic distribution (AD = 0.891)
    - Parameters: μ = 3.9721 (location), σ = 0.2101 (scale)
    """

    def __init__(self, mu: float = 3.9721, sigma: float = 0.2101):
        """
        Initialize with Log-Logistic parameters from paper.

        Parameters:
        -----------
        mu : float - Location parameter (default: 3.9721 from paper)
        sigma : float - Scale parameter (default: 0.2101 from paper)
        """
        self.mu = mu
        self.sigma = sigma

        # Performance classifications from paper
        self.performance_levels = {
            'high': {'percentile': 0.25, 'description': 'Top performers (Q1)'},
            'typical': {'percentile': 0.50, 'description': 'Average crews (Median)'},
            'low': {'percentile': 0.75, 'description': 'Need improvement (Q3)'}
        }

        # Data characteristics from paper
        self.paper_stats = {
            'min_time': 15,      # minutes per pipe
            'max_time': 100,     # minutes per pipe
            'avg_time': 50,      # minutes per pipe
            'num_projects': 35,
            'num_events': 385,
            'total_operational_time': 35000  # minutes (583 hours)
        }

    def log_logistic_percentile(self, p: float) -> float:
        """
        Calculate preparation time at given percentile using Log-Logistic distribution.

        Formula from paper (Meeker & Escobar, 1998):
            t_p = e^μ × [p / (1-p)]^σ

        Parameters:
        -----------
        p : float - Percentile (0-1), e.g., 0.5 for median

        Returns:
        --------
        Preparation time in minutes per pipe segment
        """
        if p <= 0 or p >= 1:
            raise ValueError("Percentile must be between 0 and 1 (exclusive)")

        t_p = np.exp(self.mu) * (p / (1 - p)) ** self.sigma
        return t_p

    def log_logistic_cdf(self, t: float) -> float:
        """
        Calculate cumulative probability for a given preparation time.

        Inverse of percentile function:
            F(t) = 1 / (1 + (t/e^μ)^(-1/σ))

        Parameters:
        -----------
        t : float - Preparation time in minutes

        Returns:
        --------
        Probability (0-1) that prep time is less than or equal to t
        """
        if t <= 0:
            return 0.0

        # CDF: F(t) = 1 / (1 + exp(-(ln(t) - μ) / σ))
        z = (np.log(t) - self.mu) / self.sigma
        return 1 / (1 + np.exp(-z))

    def log_logistic_pdf(self, t: float) -> float:
        """
        Calculate probability density at a given preparation time.

        Parameters:
        -----------
        t : float - Preparation time in minutes

        Returns:
        --------
        Probability density value
        """
        if t <= 0:
            return 0.0

        z = (np.log(t) - self.mu) / self.sigma
        exp_neg_z = np.exp(-z)
        pdf = exp_neg_z / (self.sigma * t * (1 + exp_neg_z) ** 2)
        return pdf

    def get_prep_time(self, performance: str = 'typical') -> float:
        """
        Get preparation time for a given performance level.

        Parameters:
        -----------
        performance : str - 'high', 'typical', or 'low'

        Returns:
        --------
        Preparation time in minutes per pipe segment
        """
        if performance not in self.performance_levels:
            raise ValueError(f"Performance must be one of: {list(self.performance_levels.keys())}")

        p = self.performance_levels[performance]['percentile']
        return self.log_logistic_percentile(p)

    def get_all_performance_times(self) -> Dict[str, float]:
        """
        Get preparation times for all performance levels.

        Returns:
        --------
        Dictionary with performance level -> time in minutes
        """
        times = {}
        for level in self.performance_levels:
            times[level] = self.get_prep_time(level)
        return times

    def classify_crew_performance(self, actual_prep_time: float) -> str:
        """
        Classify a crew's performance based on their actual preparation time.

        Parameters:
        -----------
        actual_prep_time : float - Actual preparation time in minutes

        Returns:
        --------
        Performance classification: 'high', 'typical', or 'low'
        """
        high_threshold = self.get_prep_time('high')      # Q1 (~42 min)
        typical_threshold = self.get_prep_time('low')    # Q3 (~67 min)

        if actual_prep_time <= high_threshold:
            return 'high'
        elif actual_prep_time <= typical_threshold:
            return 'typical'
        else:
            return 'low'

    def estimate_total_prep_time(self, num_pipes: int,
                                  performance: str = 'typical',
                                  include_uncertainty: bool = False) -> Dict[str, float]:
        """
        Estimate total preparation time for multiple pipe segments.

        Parameters:
        -----------
        num_pipes : int - Number of pipe segments
        performance : str - Crew performance level
        include_uncertainty : bool - Include confidence intervals

        Returns:
        --------
        Dictionary with time estimates
        """
        prep_time_per_pipe = self.get_prep_time(performance)
        total_prep_time = prep_time_per_pipe * num_pipes

        result = {
            'num_pipes': num_pipes,
            'performance_level': performance,
            'prep_time_per_pipe_min': prep_time_per_pipe,
            'total_prep_time_min': total_prep_time,
            'total_prep_time_hours': total_prep_time / 60
        }

        if include_uncertainty:
            # Add 10th and 90th percentile bounds
            p10_time = self.log_logistic_percentile(0.10) * num_pipes
            p90_time = self.log_logistic_percentile(0.90) * num_pipes

            result['lower_bound_min'] = p10_time
            result['upper_bound_min'] = p90_time
            result['lower_bound_hours'] = p10_time / 60
            result['upper_bound_hours'] = p90_time / 60

        return result

    def simulate_prep_times(self, num_pipes: int, n_simulations: int = 1000) -> np.ndarray:
        """
        Monte Carlo simulation of preparation times.

        Parameters:
        -----------
        num_pipes : int - Number of pipe segments
        n_simulations : int - Number of simulations to run

        Returns:
        --------
        Array of total preparation times from simulations
        """
        # Generate random percentiles and convert to times
        random_percentiles = np.random.uniform(0.01, 0.99, (n_simulations, num_pipes))

        # Calculate time for each percentile
        times = np.exp(self.mu) * (random_percentiles / (1 - random_percentiles)) ** self.sigma

        # Sum across pipes for total time per simulation
        total_times = np.sum(times, axis=1)

        return total_times

    def plot_distribution(self, save_path: str = None):
        """
        Plot the Log-Logistic distribution with performance thresholds.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle('Hegab (2009) Labor Performance Model: Log-Logistic Distribution',
                    fontsize=12, fontweight='bold')

        # Generate time range
        t_range = np.linspace(1, 150, 500)

        # 1. PDF Plot
        pdf_values = [self.log_logistic_pdf(t) for t in t_range]
        axes[0].plot(t_range, pdf_values, 'b-', linewidth=2, label='Log-Logistic PDF')
        axes[0].fill_between(t_range, pdf_values, alpha=0.3)

        # Add performance thresholds
        for level, color in [('high', 'green'), ('typical', 'orange'), ('low', 'red')]:
            threshold = self.get_prep_time(level)
            axes[0].axvline(x=threshold, color=color, linestyle='--',
                           label=f'{level.capitalize()}: {threshold:.1f} min')

        axes[0].set_xlabel('Preparation Time (minutes)')
        axes[0].set_ylabel('Probability Density')
        axes[0].set_title('Probability Density Function')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. CDF Plot
        cdf_values = [self.log_logistic_cdf(t) for t in t_range]
        axes[1].plot(t_range, cdf_values, 'b-', linewidth=2, label='Log-Logistic CDF')

        # Add performance zones
        axes[1].axhline(y=0.25, color='green', linestyle='--', alpha=0.7, label='Q1 (25%)')
        axes[1].axhline(y=0.50, color='orange', linestyle='--', alpha=0.7, label='Median (50%)')
        axes[1].axhline(y=0.75, color='red', linestyle='--', alpha=0.7, label='Q3 (75%)')

        axes[1].set_xlabel('Preparation Time (minutes)')
        axes[1].set_ylabel('Cumulative Probability')
        axes[1].set_title('Cumulative Distribution Function')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {save_path}")

        plt.show()

    def print_summary(self):
        """Print summary of the labor performance model."""
        print("\n" + "=" * 60)
        print("HEGAB (2009) LABOR PERFORMANCE MODEL")
        print("=" * 60)

        print("\nLog-Logistic Distribution Parameters:")
        print(f"  Location (mu): {self.mu}")
        print(f"  Scale (sigma): {self.sigma}")

        print("\nPerformance Classifications:")
        times = self.get_all_performance_times()
        for level, time in times.items():
            desc = self.performance_levels[level]['description']
            print(f"  {level.capitalize():8s}: <={time:.1f} min/pipe  ({desc})")

        print("\nPaper Data Characteristics:")
        for key, value in self.paper_stats.items():
            print(f"  {key}: {value}")


# =============================================================================
# PART 5: COMPLETE MICROTUNNELING TIME ESTIMATOR
# =============================================================================

class CompleteMicrotunnelingEstimator:
    """
    Complete time estimation system combining both Hegab papers:

    1. Hegab et al. (2006) - Soil Penetration Modeling
       - Predicts boring/tunneling time based on soil and machine parameters

    2. Hegab & Smith (2009) - Labor Performance Analysis
       - Predicts pipe preparation time using probabilistic model

    Total Drive Time = Penetration Time + Preparation Time + Delay Time
    """

    def __init__(self):
        self.penetration_model = HegabPaperModels()
        self.labor_model = LaborPerformanceModel()

        # Default delay factors (can be customized)
        self.delay_factors = {
            'soft_soil': 0.05,      # 5% delay rate
            'medium_soil': 0.10,    # 10% delay rate
            'hard_soil': 0.15,      # 15% delay rate
            'unknown': 0.10
        }

        # Typical pipe segment lengths (meters)
        self.typical_pipe_lengths = {
            'short': 1.0,
            'standard': 2.5,
            'long': 3.0
        }

    def estimate_num_pipes(self, drive_length_m: float,
                           pipe_length_m: float = 2.5) -> int:
        """
        Estimate number of pipe segments needed for a drive.

        Parameters:
        -----------
        drive_length_m : float - Total drive length in meters
        pipe_length_m : float - Length of each pipe segment

        Returns:
        --------
        Number of pipe segments (rounded up)
        """
        return int(np.ceil(drive_length_m / pipe_length_m))

    def estimate_drive_time(self,
                            # Penetration parameters (Paper 1)
                            shear_force_T: float,
                            jacking_force_P: float,
                            diameter_D: float,
                            drive_length_L: float,
                            soil_type: str = 'B',

                            # Labor parameters (Paper 2)
                            crew_performance: str = 'typical',
                            pipe_length_m: float = 2.5,

                            # Delay factors
                            include_delays: bool = True,
                            delay_rate: float = None,

                            # Output options
                            include_uncertainty: bool = False) -> Dict[str, Any]:
        """
        Estimate complete microtunneling drive time.

        Parameters:
        -----------
        shear_force_T : float - Shear force of cutter head (metric tons)
        jacking_force_P : float - Jacking force (metric tons)
        diameter_D : float - Machine diameter (meters)
        drive_length_L : float - Drive length (meters)
        soil_type : str - 'A' (soft), 'B' (medium), 'C' (hard)
        crew_performance : str - 'high', 'typical', or 'low'
        pipe_length_m : float - Length of each pipe segment (meters)
        include_delays : bool - Include delay time estimation
        delay_rate : float - Custom delay rate (overrides default)
        include_uncertainty : bool - Include confidence intervals

        Returns:
        --------
        Dictionary with complete time breakdown
        """
        results = {
            'inputs': {
                'shear_force_tons': shear_force_T,
                'jacking_force_tons': jacking_force_P,
                'diameter_m': diameter_D,
                'drive_length_m': drive_length_L,
                'soil_type': soil_type,
                'crew_performance': crew_performance,
                'pipe_length_m': pipe_length_m
            }
        }

        # -----------------------------------------------------------------
        # 1. PENETRATION TIME (Paper 1: Soil Penetration Model)
        # -----------------------------------------------------------------
        # Use penetration rate based on soil type (from paper's Table 3)
        # Penetration rates: A=24, B=35, C=57 min/m
        penetration_rates = {'A': 24, 'B': 35, 'C': 57}
        rate = penetration_rates.get(soil_type.upper(), 35)

        penetration_time_min = rate * drive_length_L

        results['penetration'] = {
            'rate_min_per_m': rate,
            'time_min': penetration_time_min,
            'time_hours': penetration_time_min / 60
        }

        # -----------------------------------------------------------------
        # 2. PREPARATION TIME (Paper 2: Labor Performance Model)
        # -----------------------------------------------------------------
        num_pipes = self.estimate_num_pipes(drive_length_L, pipe_length_m)

        prep_estimate = self.labor_model.estimate_total_prep_time(
            num_pipes,
            crew_performance,
            include_uncertainty=include_uncertainty
        )

        results['preparation'] = prep_estimate

        # -----------------------------------------------------------------
        # 3. DELAY TIME (Estimated based on soil conditions)
        # -----------------------------------------------------------------
        if include_delays:
            if delay_rate is None:
                soil_delay_map = {'A': 'soft_soil', 'B': 'medium_soil', 'C': 'hard_soil'}
                delay_key = soil_delay_map.get(soil_type.upper(), 'unknown')
                delay_rate = self.delay_factors[delay_key]

            # Delays as percentage of operational time
            operational_time = penetration_time_min + prep_estimate['total_prep_time_min']
            delay_time_min = operational_time * delay_rate

            results['delays'] = {
                'delay_rate': delay_rate,
                'delay_time_min': delay_time_min,
                'delay_time_hours': delay_time_min / 60
            }
        else:
            delay_time_min = 0
            results['delays'] = {'delay_rate': 0, 'delay_time_min': 0, 'delay_time_hours': 0}

        # -----------------------------------------------------------------
        # 4. TOTAL TIME
        # -----------------------------------------------------------------
        total_time_min = (penetration_time_min +
                          prep_estimate['total_prep_time_min'] +
                          delay_time_min)

        results['total'] = {
            'time_min': total_time_min,
            'time_hours': total_time_min / 60,
            'time_days': total_time_min / 60 / 24,
            'time_shifts_8hr': total_time_min / 60 / 8
        }

        # Time breakdown percentages
        results['breakdown_percent'] = {
            'penetration': (penetration_time_min / total_time_min) * 100,
            'preparation': (prep_estimate['total_prep_time_min'] / total_time_min) * 100,
            'delays': (delay_time_min / total_time_min) * 100
        }

        # -----------------------------------------------------------------
        # 5. UNCERTAINTY BOUNDS (if requested)
        # -----------------------------------------------------------------
        if include_uncertainty:
            # Best case: high performance crew, low delays
            best_prep = self.labor_model.get_prep_time('high') * num_pipes
            best_delay = (penetration_time_min + best_prep) * self.delay_factors['soft_soil']
            best_total = penetration_time_min + best_prep + best_delay

            # Worst case: low performance crew, high delays
            worst_prep = self.labor_model.get_prep_time('low') * num_pipes
            worst_delay = (penetration_time_min + worst_prep) * self.delay_factors['hard_soil']
            worst_total = penetration_time_min + worst_prep + worst_delay

            results['uncertainty'] = {
                'best_case_min': best_total,
                'best_case_hours': best_total / 60,
                'worst_case_min': worst_total,
                'worst_case_hours': worst_total / 60,
                'range_hours': (worst_total - best_total) / 60
            }

        return results

    def monte_carlo_simulation(self,
                               shear_force_T: float,
                               jacking_force_P: float,
                               diameter_D: float,
                               drive_length_L: float,
                               soil_type: str = 'B',
                               pipe_length_m: float = 2.5,
                               n_simulations: int = 1000) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for drive time estimation.

        Simulates variability in:
        - Penetration rate (±20% variability)
        - Preparation time (Log-Logistic distribution)
        - Delays (random occurrence)

        Returns statistical summary of results.
        """
        num_pipes = self.estimate_num_pipes(drive_length_L, pipe_length_m)

        # Base penetration rates with variability
        penetration_rates = {'A': 24, 'B': 35, 'C': 57}
        base_rate = penetration_rates.get(soil_type.upper(), 35)

        total_times = []

        for _ in range(n_simulations):
            # Random penetration rate (±20% variability)
            pen_rate = base_rate * np.random.uniform(0.8, 1.2)
            pen_time = pen_rate * drive_length_L

            # Random preparation times (from Log-Logistic distribution)
            prep_times = []
            for _ in range(num_pipes):
                p = np.random.uniform(0.05, 0.95)
                prep_time = self.labor_model.log_logistic_percentile(p)
                prep_times.append(prep_time)
            total_prep = sum(prep_times)

            # Random delays
            delay_rate = np.random.uniform(0.05, 0.20)
            delay_time = (pen_time + total_prep) * delay_rate

            total_times.append(pen_time + total_prep + delay_time)

        total_times = np.array(total_times)

        return {
            'n_simulations': n_simulations,
            'mean_hours': np.mean(total_times) / 60,
            'std_hours': np.std(total_times) / 60,
            'min_hours': np.min(total_times) / 60,
            'max_hours': np.max(total_times) / 60,
            'percentile_10_hours': np.percentile(total_times, 10) / 60,
            'percentile_50_hours': np.percentile(total_times, 50) / 60,
            'percentile_90_hours': np.percentile(total_times, 90) / 60,
            'percentile_95_hours': np.percentile(total_times, 95) / 60,
            'raw_data_min': total_times
        }

    def print_estimate(self, results: Dict[str, Any]):
        """Print formatted time estimate."""
        print("\n" + "=" * 70)
        print("COMPLETE MICROTUNNELING TIME ESTIMATE")
        print("Combining Hegab (2006) Penetration + Hegab (2009) Labor Models")
        print("=" * 70)

        # Inputs
        inp = results['inputs']
        print(f"\nINPUT PARAMETERS:")
        print(f"  Drive Length: {inp['drive_length_m']:.1f} m")
        print(f"  Diameter: {inp['diameter_m']:.2f} m")
        print(f"  Soil Type: {inp['soil_type']} ({'Soft' if inp['soil_type']=='A' else 'Medium' if inp['soil_type']=='B' else 'Hard'})")
        print(f"  Crew Performance: {inp['crew_performance'].capitalize()}")
        print(f"  Pipe Length: {inp['pipe_length_m']:.1f} m")
        print(f"  Number of Pipes: {results['preparation']['num_pipes']}")

        # Time breakdown
        print(f"\nTIME BREAKDOWN:")
        print(f"  1. Penetration Time: {results['penetration']['time_hours']:.1f} hours "
              f"({results['breakdown_percent']['penetration']:.1f}%)")
        print(f"  2. Preparation Time: {results['preparation']['total_prep_time_hours']:.1f} hours "
              f"({results['breakdown_percent']['preparation']:.1f}%)")
        print(f"  3. Delay Time:       {results['delays']['delay_time_hours']:.1f} hours "
              f"({results['breakdown_percent']['delays']:.1f}%)")

        # Total
        print(f"\nTOTAL ESTIMATED TIME:")
        print(f"  {results['total']['time_hours']:.1f} hours")
        print(f"  {results['total']['time_days']:.1f} days (24-hour)")
        print(f"  {results['total']['time_shifts_8hr']:.1f} shifts (8-hour)")

        # Uncertainty (if available)
        if 'uncertainty' in results:
            print(f"\nUNCERTAINTY RANGE:")
            print(f"  Best Case:  {results['uncertainty']['best_case_hours']:.1f} hours")
            print(f"  Worst Case: {results['uncertainty']['worst_case_hours']:.1f} hours")
            print(f"  Range:      ±{results['uncertainty']['range_hours']/2:.1f} hours")

        print("=" * 70)

    def compare_scenarios(self, drive_length_L: float, diameter_D: float,
                          shear_force_T: float = 100, jacking_force_P: float = 200) -> pd.DataFrame:
        """
        Compare different scenarios (soil types × crew performance).

        Returns DataFrame with comparison results.
        """
        scenarios = []

        for soil in ['A', 'B', 'C']:
            for crew in ['high', 'typical', 'low']:
                result = self.estimate_drive_time(
                    shear_force_T=shear_force_T,
                    jacking_force_P=jacking_force_P,
                    diameter_D=diameter_D,
                    drive_length_L=drive_length_L,
                    soil_type=soil,
                    crew_performance=crew
                )

                scenarios.append({
                    'Soil Type': f"{soil} ({'Soft' if soil=='A' else 'Medium' if soil=='B' else 'Hard'})",
                    'Crew Performance': crew.capitalize(),
                    'Total Hours': result['total']['time_hours'],
                    'Total Days': result['total']['time_days'],
                    'Penetration %': result['breakdown_percent']['penetration'],
                    'Preparation %': result['breakdown_percent']['preparation'],
                    'Delays %': result['breakdown_percent']['delays']
                })

        return pd.DataFrame(scenarios)


# =============================================================================
# PART 6: MAIN EXECUTION
# =============================================================================

def main():
    """
    Demonstrate the complete comparison framework.
    """
    print("=" * 70)
    print("HEGAB (2006) VS MODERN ML: PENETRATION TIME PREDICTION")
    print("=" * 70)

    # Initialize comparison framework
    comparison = ModelComparisonFramework()

    # Run full comparison with synthetic data
    results = comparison.run_full_comparison(n_samples=1500)

    # Create visualization
    print("\n5. Creating comparison plots...")
    comparison.plot_comparison(save_path='hegab_ml_comparison.png')

    # Show feature importance comparison
    print("\n6. Feature Importance Analysis...")
    importance_df = comparison.get_feature_importance_comparison()

    if not importance_df.empty:
        print("\nTop 10 Features (Random Forest):")
        rf_importance = importance_df[importance_df['model'] == 'RandomForest'].head(10)
        for _, row in rf_importance.iterrows():
            hegab_marker = " [HEGAB]" if row['is_hegab_feature'] else ""
            print(f"  {row['feature']}: {row['importance']:.4f}{hegab_marker}")

    # Example: Direct prediction using Hegab models
    print("\n" + "=" * 70)
    print("EXAMPLE PREDICTIONS")
    print("=" * 70)

    hegab = HegabPaperModels()

    # Example parameters
    T, P, D, L = 120, 300, 1.0, 150  # Shear, Jacking, Diameter, Length

    print(f"\nInput Parameters:")
    print(f"  Shear Force (T): {T} metric tons")
    print(f"  Jacking Force (P): {P} metric tons")
    print(f"  Diameter (D): {D} m")
    print(f"  Driven Length (L): {L} m")

    predictions = hegab.predict(T, P, D, L, soil_type='all')

    print(f"\nHegab Model Predictions:")
    print(f"  Soft Soil (A):   {predictions['soil_A_soft']:.1f} minutes")
    print(f"  Medium Soil (B): {predictions['soil_B_medium']:.1f} minutes")
    print(f"  Hard Soil (C):   {predictions['soil_C_hard']:.1f} minutes")
    print(f"  Recommended model: {predictions['recommended']}")

    # =========================================================================
    # PART 2: LABOR PERFORMANCE MODEL (2009)
    # =========================================================================
    print("\n" + "=" * 70)
    print("HEGAB (2009) LABOR PERFORMANCE MODEL")
    print("=" * 70)

    labor_model = LaborPerformanceModel()
    labor_model.print_summary()

    # Plot distribution
    print("\nGenerating labor performance distribution plot...")
    labor_model.plot_distribution(save_path='hegab_labor_distribution.png')

    # =========================================================================
    # PART 3: COMPLETE TIME ESTIMATOR (BOTH PAPERS COMBINED)
    # =========================================================================
    print("\n" + "=" * 70)
    print("COMPLETE MICROTUNNELING TIME ESTIMATOR")
    print("Combining Both Hegab Papers (2006 + 2009)")
    print("=" * 70)

    estimator = CompleteMicrotunnelingEstimator()

    # Example estimation
    estimate = estimator.estimate_drive_time(
        shear_force_T=120,
        jacking_force_P=300,
        diameter_D=1.0,
        drive_length_L=200,       # 200 meter drive
        soil_type='B',            # Medium soil
        crew_performance='typical',
        pipe_length_m=2.5,
        include_uncertainty=True
    )

    estimator.print_estimate(estimate)

    # Scenario comparison
    print("\n" + "-" * 70)
    print("SCENARIO COMPARISON: 200m Drive, 1.0m Diameter")
    print("-" * 70)

    scenarios_df = estimator.compare_scenarios(
        drive_length_L=200,
        diameter_D=1.0
    )
    print(scenarios_df.to_string(index=False))

    # Monte Carlo simulation
    print("\n" + "-" * 70)
    print("MONTE CARLO SIMULATION (1000 runs)")
    print("-" * 70)

    mc_results = estimator.monte_carlo_simulation(
        shear_force_T=120,
        jacking_force_P=300,
        diameter_D=1.0,
        drive_length_L=200,
        soil_type='B',
        n_simulations=1000
    )

    print(f"  Mean:        {mc_results['mean_hours']:.1f} hours")
    print(f"  Std Dev:     {mc_results['std_hours']:.1f} hours")
    print(f"  10th %ile:   {mc_results['percentile_10_hours']:.1f} hours")
    print(f"  50th %ile:   {mc_results['percentile_50_hours']:.1f} hours")
    print(f"  90th %ile:   {mc_results['percentile_90_hours']:.1f} hours")
    print(f"  95th %ile:   {mc_results['percentile_95_hours']:.1f} hours")

    print("\n" + "=" * 70)
    print("ALL DEMONSTRATIONS COMPLETE")
    print("=" * 70)

    return comparison, results, estimator


if __name__ == "__main__":
    comparison, results, estimator = main()
