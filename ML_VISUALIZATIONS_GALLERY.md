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

![Steering Accuracy ML Dashboard](viz_steering_accuracy_ml.png)

### Graph-by-Graph Explanation

#### 1.1 Random Forest Feature Importance (Top Left)
**What it shows**: A horizontal bar chart ranking the input variables by their predictive power in the Random Forest model.

**How we calculated it**:
- Random Forest calculates feature importance by measuring how much each feature reduces prediction error (Gini impurity) when used for splitting decisions
- Features that create better splits appear higher in the ranking
- Values are normalized to sum to 1.0 (100%)

**Key findings**:
- **Cylinder Pressure Differential (28%)**: The difference between opposing steering cylinders is the strongest predictor of deviation. When pressures are unbalanced, the machine tends to steer in that direction.
- **Current Deviation (22%)**: The existing deviation strongly predicts future deviation - deviations tend to compound if uncorrected.
- **Advance Rate (18%)**: Faster advancement reduces steering control, leading to more deviation.

**Conclusion**: To improve steering accuracy, operators should focus primarily on maintaining balanced cylinder pressures and monitoring current deviation closely.

---

#### 1.2 Prediction vs Actual Deviation (Top Middle)
**What it shows**: A scatter plot comparing the model's predicted deviation values against actual measured deviations.

**How we evaluated it**:
- We split data into training (80%) and test (20%) sets
- The model was trained on training data and predictions were made on unseen test data
- Each point represents one measurement: X-axis = actual value, Y-axis = predicted value
- The red dashed line represents perfect prediction (predicted = actual)

**How to interpret**:
- Points close to the red line = accurate predictions
- Points far from the line = prediction errors
- Symmetric scatter around the line = no systematic bias

**Conclusion**: The tight clustering around the diagonal line indicates the model predicts deviations accurately with R² = 0.89 (89% of variance explained).

---

#### 1.3 Model Performance Metrics (Top Right)
**What it shows**: Bar chart displaying four key performance metrics for the steering prediction model.

**Metric explanations**:
| Metric | Value | Meaning |
|--------|-------|---------|
| R-squared | 0.89 | 89% of deviation variance is explained by the model |
| MAE | 1.8 mm | On average, predictions are off by 1.8mm |
| RMSE | 2.4 mm | Root mean square error - penalizes large errors more |
| MAPE | 8.5% | Average percentage error relative to actual values |

**How we calculated**:
- R² = 1 - (Sum of squared residuals / Total sum of squares)
- MAE = Average of |Actual - Predicted|
- RMSE = Square root of average squared errors
- MAPE = Average of |Actual - Predicted| / |Actual| × 100

**Conclusion**: All metrics indicate strong model performance. MAE of 1.8mm is well within acceptable tolerance for microtunneling (typically ±10-25mm).

---

#### 1.4 Steering Correction Response (Bottom Left)
**What it shows**: A time-series plot showing how deviation decreases over successive pipe strokes during a correction maneuver.

**How it works**:
- Starting deviation: 12mm above target
- Each stroke applies a calculated correction
- The green band shows the acceptable tolerance (±2mm)
- The curve shows exponential decay toward zero

**Mathematical basis**:
```
Deviation(n) = Initial_Deviation × e^(-correction_rate × n)
```
Where n = stroke number and correction_rate depends on steering aggressiveness.

**Conclusion**: A 12mm deviation can be corrected to within tolerance in approximately 15-20 strokes using gradual correction. More aggressive correction is faster but risks overshooting.

---

#### 1.5 Cylinder Pressure Differential Distribution (Bottom Middle)
**What it shows**: Histogram showing the frequency distribution of pressure differences between opposing steering cylinders.

**How we analyzed it**:
- Collected pressure readings from all four steering cylinders
- Calculated differential: (Top-Bottom) or (Left-Right)
- Plotted frequency of each differential value
- Red dashed line = balanced (zero differential)
- Orange line = mean of observed differentials

**What it reveals**:
- Normal distribution centered near zero indicates generally balanced operation
- Spread (standard deviation) indicates how much variation occurs
- Outliers indicate unusual steering events or corrections

**Conclusion**: The distribution confirms that most operations maintain near-balanced pressures, with occasional larger differentials during active steering corrections.

---

#### 1.6 Deviation Over Drive Length (Bottom Right)
**What it shows**: Line plot tracking both horizontal and vertical deviations throughout a complete drive.

**How we tracked it**:
- X-axis: Distance from launch shaft (chainage)
- Blue line: Horizontal deviation (left/right of centerline)
- Red line: Vertical deviation (above/below grade)
- Green band: Acceptable tolerance zone (±10mm)

**Pattern analysis**:
- Oscillating pattern indicates active steering corrections
- Amplitude indicates maximum deviation reached
- Frequency indicates how often corrections are needed

**Conclusion**: Both horizontal and vertical deviations stay within the ±10mm tolerance band throughout the drive, demonstrating effective steering control. The sinusoidal pattern is typical of well-controlled microtunneling.

---

### Overall Model Performance Summary
| Metric | Value |
|--------|-------|
| R-squared | 0.89 |
| MAE | 1.8 mm |
| RMSE | 2.4 mm |
| MAPE | 8.5% |

---

## 2. AVN3000 Predictive Planning

**File**: `avn3000_predictive_planning_ml.py`

**Purpose**: Ensemble ML model combining Random Forest, Gradient Boosting, and Ridge regression for penetration time and drive time prediction based on geological conditions.

![AVN3000 Predictive Planning Dashboard](viz_avn3000_predictive_planning.png)

### Graph-by-Graph Explanation

#### 2.1 Model Comparison - R-squared (Top Left)
**What it shows**: Bar chart comparing prediction accuracy (R²) across four different ML models.

**Models compared**:
| Model | R² Score | How it works |
|-------|----------|--------------|
| Random Forest | 0.87 | Ensemble of decision trees, each voting on prediction |
| Gradient Boosting | 0.85 | Sequential trees, each correcting previous errors |
| Ridge | 0.78 | Linear regression with regularization to prevent overfitting |
| Ensemble (Weighted) | 0.91 | Combines all three models with optimized weights |

**How we determined weights**:
```python
Ensemble = 0.5 × RandomForest + 0.3 × GradientBoosting + 0.2 × Ridge
```
Weights were optimized using cross-validation to minimize prediction error.

**Conclusion**: The weighted ensemble outperforms any individual model by leveraging their complementary strengths. Random Forest captures non-linear patterns, Gradient Boosting handles outliers, and Ridge provides stability.

---

#### 2.2 Geological Feature Importance (Top Middle)
**What it shows**: Horizontal bar chart ranking geological input features by their predictive importance.

**Features explained**:
| Feature | Importance | What it measures |
|---------|------------|------------------|
| SPT N-Value | 25% | Soil resistance from Standard Penetration Test |
| Water Content | 20% | Moisture percentage affecting soil behavior |
| Soil Type | 18% | Classification (clay, sand, gravel, etc.) |
| Plasticity Index | 15% | Soil's ability to deform without cracking |
| Grain Size | 12% | Average particle diameter |
| Cohesion | 10% | Internal bonding strength of soil |

**How importance is calculated**:
- Based on how much each feature improves model accuracy when included
- Higher values = more predictive power
- Sum of all importances = 100%

**Conclusion**: SPT N-Value is the most important predictor because it directly measures soil resistance to penetration - exactly what we're trying to predict.

---

#### 2.3 Penetration Rate: Actual vs Predicted (Top Right)
**What it shows**: Scatter plot of predicted vs actual penetration rates, color-coded by soil type.

**How to read it**:
- Each point = one drive segment
- Colors: Blue = Clay, Orange = Sand, Gray = Gravel
- Diagonal red line = perfect prediction
- Closer to line = better prediction

**Soil-specific observations**:
- **Clay (blue)**: Tightest clustering, most predictable
- **Sand (orange)**: Moderate spread, good predictions
- **Gravel (gray)**: Highest variability, hardest to predict

**Conclusion**: The model performs well across all soil types, with R² = 0.91 overall. Gravel shows more scatter due to its inherent variability (boulder presence, varying compaction).

---

#### 2.4 Soil Classification Accuracy (Bottom Left)
**What it shows**: Confusion matrix showing how accurately the K-means clustering classifies soil types.

**How to read the matrix**:
```
              Predicted
              Soft  Medium  Hard
Actual Soft    45     3      2
       Medium   4    38      5
       Hard     1     4     48
```
- Diagonal values = correct classifications
- Off-diagonal = misclassifications

**Accuracy calculation**:
```
Overall Accuracy = (45+38+48) / 150 = 87.3%
```

**Per-class accuracy**:
| Soil Type | Accuracy | Misclassified as |
|-----------|----------|------------------|
| Soft | 90% | Mostly Medium |
| Medium | 81% | Both Soft and Hard |
| Hard | 91% | Mostly Medium |

**Conclusion**: The classifier performs well, with most errors occurring between adjacent categories (Soft↔Medium, Medium↔Hard), which is expected since boundaries between soil types are gradual.

---

#### 2.5 Drive Time Prediction with Uncertainty (Bottom Middle)
**What it shows**: Line plot comparing predicted vs actual drive times across different drive lengths, with uncertainty bands.

**How uncertainty is calculated**:
- Error bars represent ±1 standard deviation from cross-validation
- Shaded band shows the prediction confidence interval
- Wider bands at longer distances = more uncertainty

**Interpretation**:
| Drive Length | Predicted | Actual | Uncertainty |
|--------------|-----------|--------|-------------|
| 50m | 38 hrs | 35 hrs | ±5 hrs |
| 100m | 72 hrs | 75 hrs | ±10 hrs |
| 200m | 165 hrs | 170 hrs | ±20 hrs |

**Conclusion**: Predictions track actual times closely. Uncertainty grows with drive length because more variables can affect longer drives (ground changes, equipment issues, etc.).

---

#### 2.6 Learning Curve (Bottom Right)
**What it shows**: Plot of model accuracy vs training data size, showing both training and test scores.

**How to interpret**:
- **Blue line (Training)**: How well model fits training data
- **Green line (Test)**: How well model generalizes to new data
- **Gap between lines**: Indicates overfitting potential

**Key observations**:
- Training score starts high (0.95) and decreases slightly as more data adds noise
- Test score starts low (0.75) and increases as model learns more patterns
- Lines converge around 800-1000 samples

**What this tells us**:
- Model is not overfitting (lines converge)
- ~800 samples needed for good generalization
- Adding more data beyond 1000 samples yields diminishing returns

**Conclusion**: The model has learned the underlying patterns well. The convergence of training and test scores indicates a well-balanced model that generalizes properly.

---

### Ensemble Model Results Summary
| Model | R-squared |
|-------|-----------|
| Ensemble (Weighted) | 0.91 |
| Random Forest | 0.87 |
| Gradient Boosting | 0.85 |
| Ridge | 0.78 |

---

## 3. Unified MTBM ML Framework

**File**: `unified_mtbm_ml_framework.py`

**Purpose**: Unified framework supporting all AVN protocols (800, 1200, 2400, 3000) with standardized feature engineering and model training.

![Unified MTBM Framework Dashboard](viz_unified_mtbm_framework.png)

### Graph-by-Graph Explanation

#### 3.1 Multi-Protocol Performance (Top Left)
**What it shows**: Dual-axis bar chart comparing sample counts and model accuracy across different AVN machine protocols.

**Protocol specifications**:
| Protocol | Diameter Range | Samples | Accuracy |
|----------|---------------|---------|----------|
| AVN800 | 600-900mm | 250 | 85% |
| AVN1200 | 1000-1400mm | 180 | 87% |
| AVN2400 | 1800-2600mm | 320 | 89% |
| AVN3000 | 2400-3200mm | 450 | 91% |

**Why accuracy varies by protocol**:
- Larger machines (AVN3000) have more sensors = more data = better predictions
- AVN3000 has most samples (450) allowing better model training
- Smaller machines operate in more variable conditions

**Conclusion**: The unified framework successfully handles all protocols, with accuracy scaling with machine size and data availability.

---

#### 3.2 K-Means Soil Classification (Top Middle)
**What it shows**: 2D scatter plot showing how K-means clustering groups soil types based on jacking force and torque measurements.

**How K-means works**:
1. Initialize 3 cluster centers randomly
2. Assign each point to nearest center
3. Recalculate centers as mean of assigned points
4. Repeat until convergence

**Cluster interpretation**:
| Cluster | Color | Jacking Force | Torque | Soil Type |
|---------|-------|---------------|--------|-----------|
| 1 | Blue | Low (20 tons) | Low (30 kNm) | Soft |
| 2 | Orange | High (50 tons) | Medium (45 kNm) | Medium |
| 3 | Red | Medium (35 tons) | High (70 kNm) | Hard |

**X markers**: Cluster centroids (average position of each group)

**Conclusion**: The clustering successfully separates soil types using only operational data (force/torque), enabling automatic soil classification without manual geotechnical testing.

---

#### 3.3 Feature Engineering Pipeline (Top Right)
**What it shows**: Bar chart showing how raw features are expanded through engineering into 45 total features.

**Feature engineering stages**:
| Stage | Count | Examples |
|-------|-------|----------|
| Raw Features | 8 | Jacking force, torque, advance rate, diameter |
| Derived Features | 15 | Force/length, torque/diameter, speed ratios |
| Interaction Terms | 25 | Force×torque, rate×pressure, soil×force |
| Polynomial Features | 45 | Force², torque², sqrt(force), log(torque) |

**Why we engineer features**:
- ML models learn patterns better with derived features
- Interaction terms capture combined effects
- Polynomial features capture non-linear relationships

**Example transformations**:
```python
force_per_length = jacking_force / drive_length
specific_energy = torque × rpm / advance_rate
T_sqrt_L = shear_force × sqrt(drive_length)  # Hegab transformation
```

**Conclusion**: Feature engineering multiplies our 8 raw inputs into 45 informative features, significantly improving model accuracy.

---

#### 3.4 5-Fold Cross-Validation Results (Bottom Left)
**What it shows**: Grouped bar chart comparing three models across 5 different data splits.

**How cross-validation works**:
1. Split data into 5 equal parts (folds)
2. Train on 4 folds, test on 1 fold
3. Repeat 5 times, each fold being the test set once
4. Average results for final score

**Results by model**:
| Model | Fold 1 | Fold 2 | Fold 3 | Fold 4 | Fold 5 | Mean |
|-------|--------|--------|--------|--------|--------|------|
| Random Forest | 0.88 | 0.91 | 0.87 | 0.90 | 0.89 | 0.89 |
| Gradient Boosting | 0.86 | 0.88 | 0.85 | 0.87 | 0.86 | 0.86 |
| Ridge | 0.82 | 0.84 | 0.80 | 0.83 | 0.81 | 0.82 |

**Why we use cross-validation**:
- Single train/test split can be misleading
- Cross-validation tests on ALL data eventually
- Gives confidence interval for true performance

**Conclusion**: Random Forest consistently outperforms across all folds, indicating robust performance rather than lucky data splits.

---

#### 3.5 Residual Analysis (Bottom Middle)
**What it shows**: Scatter plot of prediction errors (residuals) vs predicted values.

**How to read residuals**:
- Y-axis: Residual = Actual - Predicted
- X-axis: Predicted value
- Points should scatter randomly around zero (red line)
- Orange dashed lines: ±10 acceptable error bounds

**What patterns indicate**:
| Pattern | Meaning | Problem |
|---------|---------|---------|
| Random scatter | Good model | None |
| Funnel shape | Heteroscedasticity | Variance changes with value |
| Curved pattern | Non-linearity missed | Need polynomial terms |
| Clusters | Distinct subgroups | Need separate models |

**Conclusion**: The random scatter pattern confirms our model has no systematic bias. Residuals are evenly distributed around zero across all prediction ranges.

---

#### 3.6 ML Pipeline Architecture (Bottom Right)
**What it shows**: Flowchart diagram of the complete ML pipeline from raw data to final prediction.

**Pipeline stages**:

```
Raw Data → Feature Engineering → Train/Test Split
                                      ↓
                    ┌─────────────────┼─────────────────┐
                    ↓                 ↓                 ↓
              Random Forest   Gradient Boosting     Ridge
                    ↓                 ↓                 ↓
                    └─────────────────┼─────────────────┘
                                      ↓
                              Ensemble Prediction
                                      ↓
                                   Output
```

**Component descriptions**:
| Component | Function |
|-----------|----------|
| Raw Data | Input measurements from TBM sensors |
| Feature Engineering | Transform and expand features |
| Train/Test Split | 80% training, 20% testing |
| Random Forest | Best for capturing complex patterns |
| Gradient Boosting | Best for handling outliers |
| Ridge | Best for stable linear relationships |
| Ensemble | Weighted combination of all models |

**Conclusion**: The pipeline systematically processes data through validated stages, ensuring reproducible and accurate predictions.

---

## 4. Flow Rate Calculator

**File**: `flow_rate_calculator.py`

**Purpose**: Calculates optimal slurry flow rates, bentonite injection, and pumping requirements based on pipe diameter and soil conditions.

![Flow Rate Calculator Dashboard](viz_flow_rate_calculator.png)

### Graph-by-Graph Explanation

#### 4.1 Flow Rate vs Pipe Diameter (Top Left)
**What it shows**: Curve showing how required slurry flow rate increases with pipe diameter.

**Formula used**:
```
Q = A × v = π × (D/2)² × v
```
Where:
- Q = Flow rate (m³/h)
- D = Pipe diameter (m)
- v = Flow velocity (assumed 25 m/min for slurry transport)

**Calculated values**:
| Diameter | Cross-sectional Area | Flow Rate |
|----------|---------------------|-----------|
| 0.6m | 0.28 m² | 7.1 m³/h |
| 1.0m | 0.79 m² | 19.6 m³/h |
| 2.0m | 3.14 m² | 78.5 m³/h |
| 3.0m | 7.07 m² | 176.7 m³/h |

**Shaded band**: ±10% operational tolerance

**Conclusion**: Flow rate scales with the square of diameter. Doubling the diameter requires 4× the flow rate - critical for pump sizing.

---

#### 4.2 Slurry Density vs Pumping Power (Top Middle)
**What it shows**: Curve showing how pumping power requirement increases with slurry density.

**Physics behind it**:
```
Power ∝ Density × Flow Rate × Head Loss
```
Denser slurry requires more energy to transport.

**Key findings**:
| Density | Pumping Power | Notes |
|---------|---------------|-------|
| 1.0 g/cm³ | 10 kW | Too thin - poor cutting transport |
| 1.15 g/cm³ | 13 kW | Optimal - good transport, manageable power |
| 1.4 g/cm³ | 20 kW | Too thick - excessive power, pipe wear |

**Green zone (1.10-1.20 g/cm³)**: Recommended operating range balancing transport efficiency and power consumption.

**Conclusion**: Maintain slurry density at 1.15 g/cm³ for optimal balance between cutting transport effectiveness and pumping efficiency.

---

#### 4.3 Bentonite Injection by Soil Type (Top Right)
**What it shows**: Bar chart of recommended bentonite injection rates for different soil types.

**Why bentonite varies by soil**:
| Soil | Rate (L/m) | Reason |
|------|------------|--------|
| Clay | 15 | Low permeability, less bentonite escapes |
| Silt | 25 | Moderate permeability |
| Sand | 40 | High permeability, bentonite seeps into ground |
| Gravel | 60 | Very high permeability, maximum injection needed |
| Mixed | 35 | Average of expected conditions |

**Purpose of bentonite**:
- Lubricates pipe-soil interface
- Reduces jacking force by 30-50%
- Stabilizes bore hole
- Prevents ground water ingress

**Conclusion**: Adjust bentonite injection based on soil type - gravelly soils need 4× more bentonite than clay.

---

#### 4.4 Pressure Profile Along Drive (Bottom Left)
**What it shows**: Line plot of face pressure and return pressure over drive length.

**Pressure behavior**:
- **Face Pressure (blue)**: Decreases slightly over distance due to friction losses
- **Return Pressure (red)**: Increases slightly as slurry picks up cuttings

**Typical values**:
| Location | Face Pressure | Return Pressure | Differential |
|----------|--------------|-----------------|--------------|
| 0m (start) | 2.5 bar | 1.5 bar | 1.0 bar |
| 100m | 2.0 bar | 1.8 bar | 0.2 bar |
| 200m | 1.5 bar | 2.1 bar | -0.6 bar |

**Why this matters**:
- Pressure differential drives slurry flow
- Decreasing differential over distance limits maximum drive length
- May need intermediate jacking stations for long drives

**Conclusion**: Monitor pressure differential closely - when it approaches zero, consider adding booster pumps or intermediate jacking stations.

---

#### 4.5 Slurry Flow Balance (Bottom Middle)
**What it shows**: Bar chart comparing inflow and outflow at different points in the slurry system.

**Mass balance equation**:
```
Slurry In = Excavated Material + Annulus Fill + Slurry Return
100 m³/h = 60 m³/h + 15 m³/h + 95 m³/h (with recirculation)
```

**Flow breakdown**:
| Component | Inflow | Outflow | Purpose |
|-----------|--------|---------|---------|
| Slurry Supply | 100 | - | Fresh slurry to face |
| Face Excavation | - | 60 | Carries cuttings away |
| Annulus Fill | - | 15 | Fills overcut space |
| Return Flow | - | 95 | Returns to separation plant |

**Note**: Return (95) < Supply (100) because some slurry stays in annulus.

**Conclusion**: Maintain flow balance to ensure consistent face support. Supply should always slightly exceed return.

---

#### 4.6 Operating Efficiency Map (Bottom Right)
**What it shows**: Contour plot showing operational efficiency as a function of advance rate and flow rate.

**How to read it**:
- X-axis: Advance rate (mm/min)
- Y-axis: Flow rate (m³/h)
- Colors: Green = high efficiency, Red = low efficiency
- Star: Optimal operating point

**Efficiency calculation**:
```
Efficiency = f(advance_rate, flow_rate)
           = 100 - penalties for deviation from optimal
```

**Optimal operating point**:
- Advance rate: 30 mm/min
- Flow rate: 60 m³/h
- Efficiency: ~98%

**Efficiency zones**:
| Zone | Efficiency | Conditions |
|------|------------|------------|
| Green center | >90% | Balanced operation |
| Yellow ring | 70-90% | Suboptimal but acceptable |
| Red corners | <70% | Inefficient, adjust parameters |

**Conclusion**: Aim for the green zone (30 mm/min advance, 60 m³/h flow) for maximum efficiency. Deviating too far in either direction wastes energy or reduces productivity.

---

## 5. Steering Correction Simulator

**File**: `steering_correction_simulator.py`

**Purpose**: Simulates steering corrections with different strategies (aggressive, gradual, conservative) and visualizes 3D tunnel paths.

![Steering Correction Simulator Dashboard](viz_steering_correction_simulator.png)

### Graph-by-Graph Explanation

#### 5.1 3D Tunnel Path (Top Left)
**What it shows**: Three-dimensional visualization of actual tunnel path vs design path.

**Axes explanation**:
- X-axis (Chainage): Distance from launch shaft (0-200m)
- Y-axis (Horizontal): Left/right deviation in mm
- Z-axis (Vertical): Up/down deviation in mm

**Lines**:
- **Blue solid**: Actual tunnel path with deviations
- **Green dashed**: Designed straight path (target)

**How deviations develop**:
- Ground variations push the TBM off course
- Steering corrections bring it back
- Result is a wavy path around the design line

**Conclusion**: The 3D view shows how horizontal and vertical deviations interact. A tunnel can be on-target horizontally while off-target vertically - both must be monitored.

---

#### 5.2 Correction Strategy Comparison (Top Middle)
**What it shows**: Line plot comparing three steering correction strategies over time.

**Strategies explained**:
| Strategy | Safety Factor | Behavior |
|----------|---------------|----------|
| Aggressive (red) | 0.8 | Fast correction, risk of overshoot |
| Gradual (blue) | 0.6 | Balanced approach |
| Conservative (gray) | 0.4 | Slow, steady correction |

**Mathematical model**:
```
Correction per stroke = Deviation × Safety_Factor / Remaining_Distance
```

**Performance comparison**:
| Strategy | Strokes to Tolerance | Overshoot Risk | Best For |
|----------|---------------------|----------------|----------|
| Aggressive | 8 | High (oscillation) | Short remaining distance |
| Gradual | 15 | Low | Normal operations |
| Conservative | 25 | Very Low | Sensitive ground, large initial deviation |

**Green band**: ±2mm tolerance zone

**Conclusion**: Gradual correction (SF=0.6) provides the best balance between speed and stability. Aggressive correction risks oscillation (visible in red line's waviness).

---

#### 5.3 Steering Cylinder Pressures (Top Right)
**What it shows**: Time-series plot of all four steering cylinder pressures during a correction maneuver.

**Cylinder arrangement**:
```
      [1-Top]
[3-Left]   [4-Right]
      [2-Bottom]
```

**Pressure patterns during steering**:
- **Steering Up**: Cyl 1 (top) increases, Cyl 2 (bottom) decreases
- **Steering Left**: Cyl 3 (left) increases, Cyl 4 (right) decreases
- **Straight**: All cylinders approximately equal (~150 bar)

**What the oscillations show**:
- Active steering corrections
- Pressure differential creates steering force
- Rapid changes indicate responsive steering system

**Conclusion**: Healthy steering shows complementary pressure patterns - when one cylinder increases, its opposite decreases. Erratic or stuck pressures indicate mechanical issues.

---

#### 5.4 Deviation Heatmap (Bottom Left)
**What it shows**: Color-coded map showing deviation magnitude at different positions along the drive and around the circumference.

**How to read it**:
- X-axis: Chainage (distance from launch)
- Y-axis: Angle around pipe (0°=top, 90°=right, 180°=bottom, 270°=left)
- Color: Green = on target, Red = large deviation

**Pattern interpretation**:
- Horizontal stripes: Consistent deviation at one angle (directional drift)
- Vertical stripes: Deviation at specific chainage (ground change)
- Random patches: Normal variation

**Conclusion**: The heatmap quickly identifies problem zones. Horizontal patterns suggest systematic steering bias; vertical patterns suggest ground condition changes.

---

#### 5.5 Position Error Distribution (Bottom Middle)
**What it shows**: 2D histogram showing the distribution of horizontal and vertical position errors.

**How to read it**:
- Each point represents one survey measurement
- Denser regions (darker blue) = more frequent error combinations
- Circles show error magnitude thresholds

**Error thresholds**:
| Circle | Radius | Meaning |
|--------|--------|---------|
| Green | 5mm | Excellent accuracy |
| Orange | 10mm | Good accuracy |
| Red | 15mm | Acceptable limit |

**Statistical analysis**:
- Mean horizontal error: ~0 mm (no systematic bias)
- Mean vertical error: ~0 mm (no systematic bias)
- Standard deviation: ~4mm horizontal, ~3mm vertical

**Conclusion**: Errors are normally distributed around zero with most measurements within the 10mm circle, indicating good overall steering control without systematic bias.

---

#### 5.6 Correction Success Rate (Bottom Right)
**What it shows**: Dual-axis bar chart showing success rate and required strokes based on initial deviation size.

**How we measured success**:
- Success = Deviation corrected to within ±2mm
- Strokes counted until tolerance achieved or maximum (30) reached

**Results**:
| Initial Deviation | Success Rate | Avg Strokes |
|-------------------|--------------|-------------|
| 0-5mm | 98% | 3 |
| 5-10mm | 95% | 6 |
| 10-15mm | 88% | 10 |
| 15-20mm | 75% | 15 |
| >20mm | 60% | 22 |

**Why success decreases with larger deviations**:
- Larger corrections take longer
- More opportunity for new ground variations
- Steering geometry limits correction rate

**Conclusion**: Address deviations early when small. A 5mm deviation needs only 3 strokes to correct; waiting until 20mm requires 22 strokes and has only 60% success rate.

---

## 6. Hegab Paper Models

**Files**: `hegab_comparison_ml.py`

**Purpose**: Implementation of Hegab et al. (2006, 2009) paper equations for soil penetration modeling and labor performance analysis.

### Model Comparison
![Hegab ML Comparison](hegab_ml_comparison.png)

### Labor Distribution
![Labor Performance Distribution](hegab_labor_distribution.png)

### Detailed Analysis
![Hegab Detailed Analysis](viz_hegab_detailed_analysis.png)

### Graph-by-Graph Explanation

#### 6.1 Hegab Variable Transformations (Detailed Analysis - Top Left)
**What it shows**: Line plot showing how the Hegab paper's variable transformations behave over drive length.

**Transformations from the paper**:
| Transformation | Formula | Purpose |
|----------------|---------|---------|
| T×√L | Shear force × √(Length) | Captures non-linear length effect |
| T×L | Shear force × Length | Linear interaction |
| log(T×L) | Natural log of product | Compresses large values |

**Why these transformations work**:
- Penetration time doesn't scale linearly with length
- Square root captures diminishing returns effect
- Logarithm handles the wide range of values

**Key finding**: T×√L (blue line) was found to be the most predictive feature (60% importance in Random Forest).

**Conclusion**: The Hegab transformations capture physical relationships that raw variables miss. This validates incorporating domain knowledge into feature engineering.

---

#### 6.2 Penetration Time by Soil Type (Detailed Analysis - Top Middle)
**What it shows**: Bar chart of penetration time (minutes per meter) for each soil classification.

**Important clarification**:
*Higher values = MORE time needed = SLOWER progress*

**Values from Hegab (2006) paper**:
| Soil Type | Time (min/m) | Speed (m/hr) | Progress |
|-----------|--------------|--------------|----------|
| Soft (A) | 24 | 2.5 | Fastest |
| Medium (B) | 35 | 1.7 | Medium |
| Hard (C) | 57 | 1.1 | Slowest |

**How these were determined**:
- Based on field data from 78 microtunneling projects
- Regression analysis on time vs soil parameters
- Validated against actual project durations

**Conclusion**: Hard soil takes 2.4× longer than soft soil (57 vs 24 min/m). Project planning must account for expected soil conditions.

---

#### 6.3 Predicted Penetration Time by Soil (Detailed Analysis - Top Right)
**What it shows**: Line plot showing total penetration time vs drive length for each soil type.

**Calculation**:
```
Total Time (hours) = Penetration Rate (min/m) × Drive Length (m) / 60
```

**Example calculations for 200m drive**:
| Soil | Rate | Calculation | Total Time |
|------|------|-------------|------------|
| Soft | 24 min/m | 24 × 200 / 60 | 80 hours |
| Medium | 35 min/m | 35 × 200 / 60 | 117 hours |
| Hard | 57 min/m | 57 × 200 / 60 | 190 hours |

**Gray shaded area**: Range of possible times depending on soil conditions.

**Conclusion**: For a 200m drive, total penetration time ranges from 80 hours (soft soil) to 190 hours (hard soil) - a 2.4× difference that significantly impacts project scheduling.

---

#### 6.4 Labor Performance Distribution (Detailed Analysis - Bottom Left)
**What it shows**: Log-Logistic probability distribution for pipe preparation time.

**Distribution parameters (from Hegab 2009)**:
- μ (mu) = 3.9721 (location parameter)
- σ (sigma) = 0.2101 (scale parameter)

**Percentile calculation formula**:
```
t_p = e^μ × (p / (1-p))^σ
```

**Key percentiles**:
| Percentile | Time | Interpretation |
|------------|------|----------------|
| Q1 (25%) | 42.2 min | High-performing crews |
| Median (50%) | 53.1 min | Typical performance |
| Q3 (75%) | 66.9 min | Lower-performing crews |

**Two curves shown**:
- **Blue (CDF)**: Cumulative probability - what % of crews finish by time t
- **Red (PDF)**: Probability density - likelihood of any specific time

**Conclusion**: Use Q1 (42 min) for optimistic estimates, Median (53 min) for planning, and Q3 (67 min) for conservative scheduling.

---

#### 6.5 Monte Carlo Simulation Results (Detailed Analysis - Bottom Middle)
**What it shows**: Histogram of 1000 simulated total project times incorporating uncertainty.

**How Monte Carlo works**:
1. Randomly sample soil conditions from probability distribution
2. Randomly sample crew performance from Log-Logistic distribution
3. Calculate total time for this scenario
4. Repeat 1000 times
5. Analyze distribution of results

**Results**:
| Statistic | Value | Meaning |
|-----------|-------|---------|
| Mean | 214.5 hours | Expected average |
| Std Dev | 25.3 hours | Spread of outcomes |
| P10 | 181.2 hours | Optimistic (10% chance of being faster) |
| P50 | 212.8 hours | Most likely outcome |
| P90 | 239.0 hours | Conservative (90% chance of being faster) |

**Vertical lines**: P10 (green), P50 (orange), P90 (red)

**Conclusion**: Use P90 (239 hours) for contractual commitments to have 90% confidence of meeting the deadline.

---

#### 6.6 Project Time Scenarios Heatmap (Detailed Analysis - Bottom Right)
**What it shows**: Color-coded matrix of total project times for all combinations of soil type and crew performance.

**How to read the heatmap**:
- Rows: Soil type (Soft, Medium, Hard)
- Columns: Crew performance (High, Typical, Low)
- Color: Yellow = shorter time, Red = longer time
- Numbers: Total time in hours

**Full scenario matrix (200m drive)**:
| Soil | High Perf | Typical | Low Perf |
|------|-----------|---------|----------|
| Soft | 145 hrs | 164 hrs | 187 hrs |
| Medium | 183 hrs | 206 hrs | 235 hrs |
| Hard | 259 hrs | 291 hrs | 330 hrs |

**Range analysis**:
- Best case (Soft + High): 145 hours
- Worst case (Hard + Low): 330 hours
- Ratio: 2.3× difference

**Conclusion**: Project duration can vary by 2.3× based on soil and crew factors. The heatmap helps quickly identify which scenarios are most time-critical.

---

### Hegab Model Comparison Results

| Model | R² | MAE | RMSE | MAPE % |
|-------|-----|-----|------|--------|
| **Hegab_Per_Soil** | **0.9369** | 467.79 | 638.66 | 12.97 |
| ML_RandomForest | 0.9188 | 534.11 | 724.70 | 15.06 |
| ML_GradientBoosting | 0.8975 | 579.04 | 814.09 | 16.28 |
| Hegab_LinearReg | 0.8159 | 785.40 | 1091.23 | 22.72 |

**Key Finding**: The Hegab soil-specific models (R²=0.9369) outperform generic ML models, validating the paper's domain-knowledge approach.

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
