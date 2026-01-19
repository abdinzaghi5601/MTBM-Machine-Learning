# MTBM Machine Learning - Visualization Gallery
## Comprehensive Visual Analysis of All ML Models

---

## Table of Contents
1. [Steering Accuracy ML](#1-steering-accuracy-ml)
2. [AVN3000 Predictive Planning](#2-avn3000-predictive-planning)
3. [Unified MTBM ML Framework](#3-unified-mtbm-ml-framework)
4. [Flow Rate Calculator](#4-flow-rate-calculator)
5. [Steering Correction Simulator](#5-steering-correction-simulator)
6. [Hegab Paper Models](#6-hegab-paper-models)

---

## 1. Steering Accuracy ML

**File**: `steering_accuracy_ml.py`

**Purpose**: Predicts steering deviations using Random Forest regression based on cylinder pressures, advance rates, and soil conditions.

**Key Features**:
- Feature importance analysis
- Prediction vs actual comparison
- Steering correction response curves
- Cylinder pressure distribution analysis

![Steering Accuracy ML Dashboard](viz_steering_accuracy_ml.png)

### Model Performance
| Metric | Value |
|--------|-------|
| R-squared | 0.89 |
| MAE | 1.8 mm |
| RMSE | 2.4 mm |
| MAPE | 8.5% |

### Top Features
1. Cylinder Pressure Differential (28%)
2. Current Deviation (22%)
3. Advance Rate (18%)
4. Soil Resistance (15%)

---

## 2. AVN3000 Predictive Planning

**File**: `avn3000_predictive_planning_ml.py`

**Purpose**: Ensemble ML model combining Random Forest, Gradient Boosting, and Ridge regression for penetration rate and drive time prediction.

**Key Features**:
- Multi-model ensemble comparison
- Geological feature importance
- Soil classification using K-means
- Learning curve analysis

![AVN3000 Predictive Planning Dashboard](viz_avn3000_predictive_planning.png)

### Ensemble Model Results
| Model | R-squared |
|-------|-----------|
| Ensemble (Weighted) | 0.91 |
| Random Forest | 0.87 |
| Gradient Boosting | 0.85 |
| Ridge | 0.78 |

### Geological Features
- SPT N-Value (25%)
- Water Content (20%)
- Soil Type (18%)
- Plasticity Index (15%)

---

## 3. Unified MTBM ML Framework

**File**: `unified_mtbm_ml_framework.py`

**Purpose**: Unified framework supporting all AVN protocols (800, 1200, 2400, 3000) with standardized feature engineering and model training.

**Key Features**:
- Multi-protocol support
- K-means soil classification
- Feature engineering pipeline (45 features)
- Cross-validation analysis

![Unified MTBM Framework Dashboard](viz_unified_mtbm_framework.png)

### Protocol Performance
| Protocol | Samples | Accuracy |
|----------|---------|----------|
| AVN800 | 250 | 85% |
| AVN1200 | 180 | 87% |
| AVN2400 | 320 | 89% |
| AVN3000 | 450 | 91% |

### Feature Engineering Pipeline
- Raw Features: 8
- Derived Features: 15
- Interaction Terms: 25
- Polynomial Features: 45 (total)

---

## 4. Flow Rate Calculator

**File**: `flow_rate_calculator.py`

**Purpose**: Calculates optimal slurry flow rates, bentonite injection, and pumping requirements based on pipe diameter and soil conditions.

**Key Features**:
- Flow rate vs diameter curves
- Slurry density optimization
- Pressure profile along drive
- Operating efficiency maps

![Flow Rate Calculator Dashboard](viz_flow_rate_calculator.png)

### Key Parameters
| Parameter | Optimal Range |
|-----------|---------------|
| Slurry Density | 1.10-1.20 g/cm3 |
| Optimal Point | 1.15 g/cm3 |
| Flow Balance | 95% return efficiency |

### Bentonite Injection by Soil
| Soil Type | Injection Rate |
|-----------|----------------|
| Clay | 15 L/m |
| Silt | 25 L/m |
| Sand | 40 L/m |
| Gravel | 60 L/m |

---

## 5. Steering Correction Simulator

**File**: `steering_correction_simulator.py`

**Purpose**: Simulates steering corrections with different strategies (aggressive, gradual, conservative) and visualizes 3D tunnel paths.

**Key Features**:
- 3D tunnel path visualization
- Correction strategy comparison
- Cylinder pressure response
- Position error distribution

![Steering Correction Simulator Dashboard](viz_steering_correction_simulator.png)

### Correction Strategies
| Strategy | Safety Factor | Best For |
|----------|---------------|----------|
| Aggressive | 0.8 | Small deviations, short remaining distance |
| Gradual | 0.6 | Normal operations |
| Conservative | 0.4 | Large deviations, sensitive ground |

### Success Rates by Initial Deviation
| Initial Deviation | Success Rate | Avg Strokes |
|-------------------|--------------|-------------|
| 0-5mm | 98% | 3 |
| 5-10mm | 95% | 6 |
| 10-15mm | 88% | 10 |
| 15-20mm | 75% | 15 |
| >20mm | 60% | 22 |

---

## 6. Hegab Paper Models

**Files**: `hegab_comparison_ml.py`

**Purpose**: Implementation of Hegab et al. (2006, 2009) paper equations for soil penetration modeling and labor performance analysis.

**Key Features**:
- Variable transformations (T*sqrt(L), log(TL))
- Soil-specific penetration rates
- Log-Logistic labor distribution
- Monte Carlo simulation

### Model Comparison
![Hegab ML Comparison](hegab_ml_comparison.png)

### Labor Distribution
![Labor Performance Distribution](hegab_labor_distribution.png)

### Detailed Analysis
![Hegab Detailed Analysis](viz_hegab_detailed_analysis.png)

### Penetration Time by Soil Type (Hegab 2006)
*Higher time = Slower progress (harder ground takes longer)*

| Soil Type | Time (min/m) | Speed Equivalent | Progress |
|-----------|--------------|------------------|----------|
| Soft (A) | 24 | 2.5 m/hour | Fastest |
| Medium (B) | 35 | 1.7 m/hour | Medium |
| Hard (C) | 57 | 1.1 m/hour | Slowest |

### Labor Performance (Hegab 2009)
| Performance | Prep Time |
|-------------|-----------|
| High (Q1) | <= 42.2 min |
| Typical (Median) | <= 53.1 min |
| Low (Q3) | <= 66.9 min |

### Monte Carlo Results (200m drive)
| Metric | Value |
|--------|-------|
| Mean | 214.5 hours |
| Std Dev | 25.3 hours |
| P90 | 239.0 hours |

---

## How to Regenerate Visualizations

Run the visualization generator script:

```bash
python generate_all_visualizations.py
```

This will regenerate all 6 visualization files:
1. `viz_steering_accuracy_ml.png`
2. `viz_avn3000_predictive_planning.png`
3. `viz_unified_mtbm_framework.png`
4. `viz_flow_rate_calculator.png`
5. `viz_steering_correction_simulator.png`
6. `viz_hegab_detailed_analysis.png`

---

## Quick Links to Source Code

| Visualization | Source File |
|---------------|-------------|
| Steering Accuracy | [steering_accuracy_ml.py](steering_accuracy_ml.py) |
| AVN3000 Planning | [avn3000_predictive_planning_ml.py](avn3000_predictive_planning_ml.py) |
| Unified Framework | [unified_mtbm_ml_framework.py](unified_mtbm_ml_framework.py) |
| Flow Rate | [flow_rate_calculator.py](flow_rate_calculator.py) |
| Steering Simulator | [steering_correction_simulator.py](steering_correction_simulator.py) |
| Hegab Models | [hegab_comparison_ml.py](hegab_comparison_ml.py) |

---

*Generated: January 19, 2026*
*MTBM Machine Learning Project*
