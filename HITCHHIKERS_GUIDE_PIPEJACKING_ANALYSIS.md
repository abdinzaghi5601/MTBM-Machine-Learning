# Hitchhiker's Guide to Pipejacking - Excel Calculator Analysis

**Author:** Lutz Henke  
**File:** `Hitchhiker's Guide to Pipejacking.xls`  
**Structure:** 145 rows × 14 columns

---

## Overview

This Excel calculator performs calculations for **pipejacking operations**, including:
1. **Push Power & Jacking Speed** calculations
2. **Slurry Flow Rates** for different velocities
3. **Power Requirements** for all equipment
4. **Cable Sizing** calculations

---

## Input Parameters (White Squares - User Input)

### Main Jacking Station Parameters
| Parameter | Location | Unit | Example Value |
|-----------|----------|------|---------------|
| Length of Jacking pipe | Row 8, Col E | Metre | 3 |
| Main Jacking station No. of cylinders | Row 12, Col E | - | 4 |
| Main Jacking station dia. of cylinders | Row 14, Col E | mm | 285 |
| Main Jacking station push oil available | Row 16, Col E | L/min | 60 |
| Main Jacking station max. oil pressure | Row 18, Col E | Bar | 500 |

### Interjack Station Parameters
| Parameter | Location | Unit | Example Value |
|-----------|----------|------|---------------|
| Interjack station No. of cylinders | Row 20, Col E | No. | 16 |
| Interjack station dia. of cylinders | Row 22, Col E | mm | 140 |
| Interjack station push oil available | Row 24, Col E | L/min | 60 |
| Interjack maximum oil pressure | Row 26, Col E | Bar | 500 |

### Slurry System Parameters
| Parameter | Location | Unit | Example Value |
|-----------|----------|------|---------------|
| Slurry pipe diameter | Row 10, Col E | mm | 150 |

### Power System Parameters
| Parameter | Location | Unit | Example Value |
|-----------|----------|------|---------------|
| Power Factor | Row 22, Col L | PF | 0.8 |
| Required Voltage | Row 24, Col L | Voltage | 400 |

---

## Calculations Performed

### 1. Main Jacking Station Calculations

#### Total Force Calculation
**Formula:**
```
Force = Pressure × Area × Number of Cylinders
```

**Implementation:**
- **Combined full bore Area** (Row 38, Col E):
  ```
  Area = π × (diameter/2)² × Number of Cylinders
  Area = π × (285/2)² × 4 = 2552.0895 Sq. Cm
  ```

- **Total Force** (Row 40, Col E):
  ```
  Force = (Pressure × Area) / 1000
  Force = (500 bar × 2552.0895 cm²) / 1000 = 1276.04475 Tonnes
  ```
  *Note: 1 bar = 1 kg/cm², converted to tonnes*

#### Maximum Jacking Speed
**Formula:**
```
Speed = (Oil Flow Rate) / (Cylinder Area)
```

**Implementation:**
- **Max. Speed of Push** (Row 42, Col E):
  ```
  Speed = (60 L/min) / (2552.0895 cm²)
  Speed = 23.510147273440058 Cm. Per Minute
  ```

#### Time to Push One Pipe
**Formula:**
```
Time = Pipe Length / Speed
```

**Implementation:**
- **Max. speed to push a pipe** (Row 44, Col E):
  ```
  Time = 300 cm / 23.510147273440058 cm/min
  Time = 12.7604475 minutes
  ```

---

### 2. Interjack Station Calculations

Similar calculations as Main Jacking Station but with different parameters:

- **Combined full bore Area** (Row 50, Col E):
  ```
  Area = π × (140/2)² × 16 = 2463.328 Sq. Cm
  ```

- **Total Force** (Row 52, Col E):
  ```
  Force = (500 bar × 2463.328 cm²) / 1000 = 1231.664 Tonnes
  ```

---

### 3. Slurry System Flow Rate Calculations

**Base Formula:**
```
Flow Rate = Cross-sectional Area × Velocity
```

**Cross-sectional Area Calculation:**
```
Area = π × (diameter/2)²
Area = π × (150 mm / 2)² = π × 75² = 17,673.75 mm²
Area = 0.01767375 m²
```

**Volume per Minute:**
```
Volume (L/min) = Area (m²) × Velocity (m/s) × 60 (s/min) × 1000 (L/m³)
Volume = 0.01767375 × velocity × 60 × 1000
Volume = 17.67375 × velocity L/min
```

**Flow Rate Table (Cubic M/hr):**

| Velocity (m/s) | Flow Rate (m³/hr) | Conditions |
|----------------|-------------------|------------|
| 3.0 | 190.88 | Sandy Conditions |
| 3.25 | 206.78 | - |
| 3.5 | 222.69 | Larger & Irregular in shape |
| 3.75 | 238.60 | - |
| 4.0 | 254.50 | - |
| 4.25 | 270.41 | - |
| 4.5 | 286.31 | - |
| 4.75 | 302.22 | - |
| 5.0 | 318.13 | - |

**Formula:**
```
Flow Rate (m³/hr) = 17.67375 × velocity (m/s) × 60 / 1000
Flow Rate (m³/hr) = 1.060425 × velocity (m/s)
```

**Notes from Calculator:**
- In sandy conditions: velocity of 3.0 m/s may be sufficient
- For larger & irregular material: velocity may need to be increased to 3.5 m/s
- Bentonite density also affects required velocity
- Discharge density: typically 1.25
- Feed line density: typically 1.08

---

### 4. Power Requirements

#### Motor Power List
| Motor | Power (kW) | Equipment |
|-------|------------|-----------|
| Motor No 1 | 30 | Machine |
| Motor No 2 | 110 | Container |
| Motor No 3 | 30 | Separation Plant |
| Motor No 4 | 30 | Bentonite Plant |
| Motor No 5 | 30 | Office/Fitting Shop |
| Motor No 6 | 250 | Main Shaft Pump |
| Motor No 7 | 175 | Duty Shaft Pump |
| Motor No 8 | 0 | - |
| Motor No 9 | 0 | - |
| Motor No 10 | 0 | - |

#### Total Power Calculation
**Total Power (kW):** Row 26, Col L = 739 kW

**Total Power (kVA):** Row 28, Col L = 923.75 kVA
```
kVA = kW / Power Factor
kVA = 739 / 0.8 = 923.75 kVA
```

**Generator Sizing:**
- Main Generator: 739 kVA (80% of 923.75 kVA)
- Weekend Generator: 100 kVA

**Note:** Generator calculation is based on all motors on full load. 80% of main generator load would be sufficient.

---

### 5. Cable Sizing Calculations

**Reference Standards:**
- BS5467 & BS7211
- Table 4E2A
- Multi-core cable with thermosetting insulation, non-armoured
- COPPER CONDUCTORS

**Cable Sizing Table:**

| Cross Section (mm²) | Current Rating (Amps) |
|---------------------|----------------------|
| 1 | 18 |
| 1.5 | 23 |
| 2.5 | 32 |
| 4 | 42 |
| 6 | 54 |
| 10 | 75 |
| 16 | 100 |
| 25 | 127 |
| 35 | 158 |
| 50 | 192 |
| 70 | 246 |
| 95 | 298 |
| 120 | 346 |
| 150 | 399 |
| 185 | 456 |
| 240 | 538 |
| 300 | 621 |
| 400 | 741 |

**Amperage Calculation:**
```
Amps = (kVA × 1000) / (Voltage × √3)
```

**Examples:**
- **Container M (545):** 655 kVA → 2832.875 Amps (at 400V)
- **Separation Plant:** 50 kVA → 216.25 Amps (at 400V)
- **Other equipment:** 9.5 kVA → 39.60 Amps (at 415V)

**Formula:**
```
For 3-phase: Amps = (kVA × 1000) / (Voltage × 1.732)
For single-phase: Amps = (kVA × 1000) / Voltage
```

---

## Key Formulas Summary

### Jacking Force
```
Force (Tonnes) = (Pressure (bar) × Area (cm²)) / 1000
Area (cm²) = π × (diameter/2)² × Number of Cylinders
```

### Jacking Speed
```
Speed (cm/min) = (Oil Flow (L/min) × 1000) / Area (cm²)
```

### Time to Push Pipe
```
Time (min) = Pipe Length (cm) / Speed (cm/min)
```

### Slurry Flow Rate
```
Flow Rate (m³/hr) = π × (diameter/2)² × velocity (m/s) × 3600
Flow Rate (m³/hr) = 1.060425 × velocity (m/s)  [for 150mm pipe]
```

### Power Calculations
```
kVA = kW / Power Factor
Amps (3-phase) = (kVA × 1000) / (Voltage × 1.732)
```

---

## Usage Instructions

1. **Input Parameters:** Enter values in white squares (cells marked for input)
2. **Review Calculations:** All calculated values appear automatically
3. **Check Results:**
   - Main Jacking Station force and speed
   - Interjack Station force
   - Slurry flow rates for different velocities
   - Total power requirements
   - Cable sizing recommendations

---

## Important Notes

1. **Slurry Velocity Selection:**
   - Sandy conditions: 3.0 m/s minimum
   - Larger/irregular material: 3.5 m/s recommended
   - Bentonite density affects required velocity

2. **Power Factor:**
   - Default: 0.8
   - Used to convert kW to kVA

3. **Generator Sizing:**
   - Based on all motors at full load
   - 80% of calculated load is typically sufficient

4. **Cable Sizing:**
   - Use Table 4E2A for current ratings
   - Select cable size based on calculated amperage
   - Consider derating factors for installation conditions

---

## File Structure

- **Rows 1-30:** Title, author, input parameters, motor list, power totals
- **Rows 31-45:** Main Jacking Station calculations
- **Rows 46-52:** Interjack Station calculations
- **Rows 55-87:** Slurry system flow rate calculations and notes
- **Rows 88-116:** Site power requirements and generator sizing
- **Rows 119-145:** Cable sizing table and calculations

---

*Analysis completed: 2024*  
*Based on Excel file: "Hitchhiker's Guide to Pipejacking.xls" by Lutz Henke*

