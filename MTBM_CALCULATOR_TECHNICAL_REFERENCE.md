# MTBM Unified Calculator - Technical Reference

## Complete Formula Documentation for Microtunneling & Pipejacking Calculations

**Version:** 1.0
**Last Updated:** January 2026
**Authors:** Developed with Claude AI assistance

---

## Table of Contents

1. [Overview](#overview)
2. [Module 1: Slurry Properties Calculator](#module-1-slurry-properties-calculator)
3. [Module 2: Slurry Flow Calculator](#module-2-slurry-flow-calculator)
4. [Module 3: Pipejacking Calculator](#module-3-pipejacking-calculator)
5. [Module 4: Advanced Pipejacking Calculations](#module-4-advanced-pipejacking-calculations)
6. [Quick Reference Tables](#quick-reference-tables)
7. [References](#references)

---

## Overview

The MTBM Unified Calculator is a comprehensive command-line tool for microtunneling and pipejacking engineering calculations. It consolidates formulas from multiple industry sources into a single, validated calculation suite.

### Source References

- **Excellence Pump Industry** - Slurry Pump Manual
- **Hitchhiker's Guide to Pipejacking** - Lutz Henke (Herrenknecht Academy)
- **Duhme et al.** - Excavation mass balance methodology
- **Durand-Condolios** - Critical deposition velocity
- **German Pipejacking Exercise Examples** - Practical worked solutions

### Running the Calculator

```bash
# Interactive mode
python mtbm_calculator_cli.py

# Quick calculations
python mtbm_calculator_cli.py --quick

# Command-line calculations
python mtbm_calculator_cli.py --calc <type> <args>
```

---

## Module 1: Slurry Properties Calculator

### Purpose
Calculate the fundamental relationships between slurry properties: specific gravity, volume concentration, and weight concentration.

### Variables

| Symbol | Name | Description | Typical Range |
|--------|------|-------------|---------------|
| S | Specific Gravity of Solids | SG of dry solid particles | 2.4 - 2.8 (soil), up to 4.5 (ore) |
| Sw | Specific Gravity of Liquid | SG of carrier fluid | 1.0 (water), 1.02-1.05 (bentonite) |
| Sm | Specific Gravity of Mixture | SG of slurry mixture | 1.1 - 1.8 typical |
| Cv | Volume Concentration | Solids by volume (decimal) | 0.05 - 0.35 |
| Cw | Weight Concentration | Solids by weight (decimal) | 0.10 - 0.60 |

### Formulas

#### 1.1 Mixture Specific Gravity from Volume Concentration

$$S_m = S_w + C_v (S - S_w)$$

**Description:** Calculates the specific gravity of the slurry mixture based on the volume fraction of solids.

**Example:**
- S = 2.65 (silica sand)
- Sw = 1.0 (water)
- Cv = 0.25 (25% by volume)
- Sm = 1.0 + 0.25 × (2.65 - 1.0) = **1.4125**

---

#### 1.2 Volume Concentration from Mixture SG

$$C_v = \frac{S_m - S_w}{S - S_w}$$

**Description:** Back-calculates volume concentration when mixture SG is measured (e.g., with a densitometer).

**Example:**
- Measured Sm = 1.30
- Sw = 1.0, S = 2.65
- Cv = (1.30 - 1.0) / (2.65 - 1.0) = **0.182 (18.2%)**

---

#### 1.3 Weight Concentration from Volume Concentration

$$C_w = \frac{S \times C_v}{S_m}$$

**Description:** Converts volume concentration to weight concentration. Weight concentration is often preferred for material handling calculations.

---

#### 1.4 Mixture SG from Weight Concentration

$$S_m = \frac{S_w}{1 - C_w \left(1 - \frac{S_w}{S}\right)}$$

**Description:** Alternative calculation when weight concentration is known from sampling.

---

#### 1.5 Fundamental Relationship

$$\frac{C_w}{C_v} = \frac{S}{S_m}$$

**Description:** This relationship is always true and independent of the carrier fluid. Useful for cross-checking calculations.

---

## Module 2: Slurry Flow Calculator

### Purpose
Calculate slurry transport requirements including flow rates, velocities, settling characteristics, and system sizing.

### 2.1 Excavation Rate

#### Formula

$$V_{exc} = A_{cutter} \times v_{advance}$$

Where:
- $A_{cutter} = \frac{\pi \times D_{cutter}^2}{4}$ (cutter face area, m²)
- $v_{advance}$ = advance rate (m/hr)
- $V_{exc}$ = excavation rate (m³/hr)

**Description:** Calculates the in-situ volume of ground excavated per hour.

---

### 2.2 Bulked Volume (with Swell Factor)

#### Formula

$$V_{bulked} = V_{insitu} \times f_{swell}$$

| Soil Type | Swell Factor |
|-----------|--------------|
| Clay | 1.20 - 1.35 |
| Silt | 1.15 - 1.25 |
| Sand | 1.10 - 1.20 |
| Gravel | 1.15 - 1.25 |
| Rock | 1.40 - 1.80 |

**Description:** Ground expands when excavated. The swell factor accounts for this volume increase in slurry transport.

---

### 2.3 Settling Velocity (Stokes' Law)

#### Formula - Laminar Flow (Re < 1)

$$v_s = \frac{g \times d^2 \times (S - S_w)}{18 \times \nu}$$

#### Formula - Transitional Flow (1 < Re < 1000)

$$v_s = \sqrt{\frac{4 \times g \times d \times (S - S_w)}{3 \times C_D}}$$

With drag coefficient (Schiller-Naumann):
$$C_D = \frac{24}{Re}\left(1 + 0.15 \times Re^{0.687}\right)$$

Where:
- $v_s$ = settling velocity (m/s)
- $g$ = gravity (9.81 m/s²)
- $d$ = particle diameter (m)
- $\nu$ = kinematic viscosity (1.0×10⁻⁶ m²/s for water)
- $Re$ = Reynolds number = $\frac{v_s \times d}{\nu}$

**Description:** Determines how fast particles settle in still fluid. Critical for understanding whether particles will deposit in pipelines.

**Flow Regime Classification:**
| Reynolds Number | Regime | Method |
|-----------------|--------|--------|
| Re < 1 | Laminar (Stokes) | Direct formula |
| 1 < Re < 1000 | Transitional | Iterative with drag correction |
| Re > 1000 | Turbulent | Newton's law |

---

### 2.4 Critical Deposition Velocity (Durand-Condolios)

#### Formula

$$V_c = F_L \times \sqrt{2 \times g \times D \times (S - 1)}$$

Where:
- $V_c$ = critical velocity (m/s)
- $F_L$ = Durand factor (dimensionless, typically 0.9 - 1.5)
- $D$ = pipe inner diameter (m)
- $S$ = particle specific gravity

#### Durand Factor Estimation

$$F_L = 1.3 \times C_v^{0.125} \times \left(1 - e^{-6.9 \times d_{50}}\right)$$

**Description:** The minimum velocity required to keep particles suspended and prevent deposition in the pipeline. Operating below this velocity causes settling, blockages, and increased wear.

**Typical Critical Velocities:**
| Pipe Size | Sand (d50=0.5mm) | Gravel (d50=5mm) |
|-----------|------------------|------------------|
| DN150 | 2.8 - 3.2 m/s | 3.5 - 4.0 m/s |
| DN200 | 3.0 - 3.5 m/s | 3.8 - 4.3 m/s |
| DN250 | 3.2 - 3.7 m/s | 4.0 - 4.5 m/s |

---

### 2.5 Alternative: Wasp Critical Velocity

#### Formula

$$V_c = 3.5 \times \sqrt{2 \times g \times D \times (S - 1)} \times \left(\frac{d_{50}}{D}\right)^{0.167}$$

**Description:** Alternative method that explicitly includes particle size ratio. Often gives more conservative (higher) values.

---

### 2.6 Recommended Minimum Velocity

$$V_{min} = 1.2 \times \max(V_{c,Durand}, V_{c,Wasp})$$

**Description:** A 20% safety factor is applied to the higher of the two calculated critical velocities.

**Operating Velocity Guidelines:**
| Condition | Velocity Range |
|-----------|----------------|
| Minimum (fine sand) | 3.0 m/s |
| Normal operation | 3.5 - 4.0 m/s |
| Coarse/irregular | 4.0 - 4.5 m/s |
| Maximum recommended | 5.0 m/s |

---

### 2.7 Slurry Flow Rate from Velocity

#### Formula

$$Q = V \times A = V \times \frac{\pi \times D^2}{4}$$

Converting units:
$$Q_{m^3/hr} = V_{m/s} \times A_{m^2} \times 3600$$

**Description:** Calculates required pump flow rate to achieve target velocity.

---

### 2.8 Excavation Mass Balance (Duhme Method)

#### Formula

$$\dot{m}_{exc} = Q_{out} \times \rho_{out} - Q_{in} \times \rho_{in}$$

$$V_{exc} = \frac{\dot{m}_{exc}}{\rho_{solid}}$$

Where:
- $Q_{in}$, $Q_{out}$ = feed and return flow rates (m³/hr)
- $\rho_{in}$, $\rho_{out}$ = feed and return densities (kg/m³)
- $\dot{m}_{exc}$ = excavated mass rate (kg/hr)

**Description:** Real-time monitoring of excavation rate using flow meters and densitometers in the slurry circuit. Enables detection of over-excavation or face collapse.

---

### 2.9 Slurry Head Loss (Darcy-Weisbach with Durand Correction)

#### Water Head Loss

$$h_{water} = f \times \frac{L}{D} \times \frac{V^2}{2g}$$

Where friction factor $f$ from Swamee-Jain:
$$f = \frac{0.25}{\left[\log_{10}\left(\frac{\varepsilon}{3.7D} + \frac{5.74}{Re^{0.9}}\right)\right]^2}$$

#### Slurry Correction Factor (Durand)

$$\phi = 82 \times C_v \times \sqrt{\frac{g \times D \times (S-1)}{V^2}} \times \left(\frac{V}{\sqrt{g \times d_{50}}}\right)^{-1.5}$$

#### Total Slurry Head Loss

$$h_{slurry} = h_{water} \times (1 + \phi) \times \frac{S_m}{S_w}$$

**Description:** Slurry creates additional head loss compared to clear water due to particle-fluid interactions. The Durand correction factor accounts for this.

---

### 2.10 Bentonite Injection Requirements

#### Formula

$$Q_{bentonite} = \frac{V_{annulus/m} \times v_{advance} \times f_{loss}}{1000}$$

Where:
- $V_{annulus/m}$ = annular volume per meter (L/m)
- $v_{advance}$ = advance rate (m/hr)
- $f_{loss}$ = ground loss factor (1.1 - 1.3 typical)

**Bentonite Injection Rates by Ground Type:**
| Ground Type | Rate (L/m²) | Description |
|-------------|-------------|-------------|
| Clay | 5 - 10 | Low permeability, minimal loss |
| Silt | 10 - 15 | Moderate loss |
| Sand | 15 - 25 | Higher permeability |
| Gravel | 25 - 40 | High loss, may need polymer |
| Mixed | 15 - 25 | Variable conditions |
| Rock | 5 - 15 | Fracture dependent |

---

## Module 3: Pipejacking Calculator

### Purpose
Calculate jacking forces, speeds, power requirements, and electrical sizing for pipejacking operations.

### 3.1 Jacking Force

#### Formula

$$F = P \times A_{total}$$

Where:
$$A_{total} = n \times \frac{\pi \times D_{cyl}^2}{4}$$

- $F$ = total force (kN)
- $P$ = hydraulic pressure (bar = 0.1 N/mm²)
- $n$ = number of cylinders
- $D_{cyl}$ = cylinder bore diameter (mm)

**Unit Conversion:**
$$F_{tonnes} = \frac{P_{bar} \times A_{cm^2}}{1000}$$

**Description:** Calculates the maximum thrust force available from the jacking station. This must exceed the total friction resistance of the pipe string.

**Example:**
- 4 cylinders × 285mm diameter at 500 bar
- Area = 4 × π × (28.5/2)² = 2551 cm²
- Force = 500 × 2551 / 1000 = **1276 tonnes**

---

### 3.2 Jacking Speed

#### Formula

$$v = \frac{Q_{oil}}{A_{total}}$$

Where:
- $v$ = advance speed (cm/min)
- $Q_{oil}$ = oil flow rate (L/min)
- $A_{total}$ = combined cylinder area (cm²)

**Description:** Calculates maximum jacking speed based on hydraulic pump capacity.

**Example:**
- Flow = 60 L/min, Area = 2551 cm²
- Speed = 60 × 1000 / 2551 = **23.5 cm/min**

---

### 3.3 Power Requirements

#### Formula

$$P_{kVA} = \frac{P_{kW}}{pf}$$

Where:
- $P_{kW}$ = sum of all motor powers
- $pf$ = power factor (typically 0.8)

#### Generator Sizing

$$P_{generator} = \frac{P_{kVA}}{0.8}$$

**Description:** Generators should be sized at 80% loading for optimal efficiency and to handle starting currents.

---

### 3.4 Cable Amperage

#### Formula (3-phase)

$$I = \frac{P_{kVA} \times 1000}{\sqrt{3} \times V}$$

#### Formula (1-phase)

$$I = \frac{P_{kVA} \times 1000}{V}$$

**Description:** Calculates current draw for cable sizing.

**Cable Sizing Table (BS5467/BS7211):**
| Current (A) | Cable Size (mm²) |
|-------------|------------------|
| ≤ 32 | 6 |
| ≤ 43 | 10 |
| ≤ 59 | 16 |
| ≤ 77 | 25 |
| ≤ 100 | 35 |
| ≤ 127 | 50 |
| ≤ 158 | 70 |
| ≤ 190 | 95 |
| ≤ 232 | 120 |
| ≤ 269 | 150 |
| ≤ 313 | 185 |
| ≤ 380 | 240 |

---

## Module 4: Advanced Pipejacking Calculations

### Purpose
Comprehensive calculations for monitoring, optimization, and troubleshooting pipejacking operations based on German industry exercise examples.

---

### 4.1 Overcut Volume

#### Formula

$$V_{overcut} = \frac{\pi}{4} \times (D_{cutter}^2 - D_{pipe}^2) \times L$$

Or per meter:
$$V_{overcut/m} = \frac{\pi}{4} \times (D_{cutter}^2 - D_{pipe}^2)$$

**Description:** Calculates the annular void between the cutterhead and pipe that must be filled with lubrication (bentonite).

**Example:**
- Cutter diameter = 2560 mm
- Pipe outer diameter = 2500 mm
- Overcut per side = (2560 - 2500) / 2 = **30 mm**
- Volume = π/4 × (2.56² - 2.50²) = **0.238 m³/m**

---

### 4.2 Bentonite Flow for Overcut Filling

#### Formula

$$Q_{bento} = \frac{V_{overcut/m} \times v_{advance} \times f_{loss}}{t_{per\_meter}}$$

Where:
- $t_{per\_meter} = \frac{100}{v_{advance}}$ (minutes per meter at cm/min advance)
- $f_{loss}$ = ground loss factor (typically 1.10 - 1.30)

**Description:** Calculates the bentonite pumping rate required to fill the annular space as the machine advances.

**Example:**
- Overcut volume = 0.238 m³/m = 238 L/m
- Advance = 10 cm/min → 10 min/m
- Base flow = 238 / 10 = 23.8 L/min
- With 15% loss: 23.8 × 1.15 = **27.4 L/min**

---

### 4.3 Lubrication Station Spacing

#### Formula

$$n_{stations} = \frac{L_{tunnel} - L_{machine}}{L_{spacing}} + 1$$

Where:
- $L_{spacing} = n_{pipes} \times L_{pipe}$ (spacing between stations)

#### Total Lubrication Volume

$$V_{total} = n_{stations} \times V_{per\_station}$$

**Description:** Calculates number and placement of intermediate lubrication stations along the pipeline for friction reduction on long drives.

**Typical Configurations:**
| Drive Length | Station Spacing | Notes |
|--------------|-----------------|-------|
| < 100m | None needed | Rely on machine injection |
| 100-200m | Every 30-50m | 1-2 intermediate stations |
| 200-500m | Every 15-25m | Multiple stations |
| > 500m | Every 10-20m | Continuous lubrication critical |

---

### 4.4 Excavation Volume with Separation Efficiency

#### Formula

$$V_{separated} = V_{bulked} \times \eta_{sep}$$

$$V_{loss} = V_{bulked} \times (1 - \eta_{sep})$$

Where:
- $\eta_{sep}$ = separation efficiency (typically 0.90 - 0.98)

**Description:** Not all excavated material is recovered by the separation plant. Some fine particles remain in the slurry circuit.

**Typical Separation Efficiencies:**
| Equipment | Efficiency |
|-----------|------------|
| Shaker screens only | 80-85% |
| Shakers + hydrocyclones | 90-95% |
| Full separation plant | 95-98% |

---

### 4.5 Pipe Buoyancy Check

#### Formula

$$F_{buoyancy} = M_{water} - M_{pipe}$$

Where:
$$M_{water} = \frac{\pi \times D_{outer}^2}{4} \times L \times \rho_{water}$$

$$M_{pipe} = \frac{\pi \times (D_{outer}^2 - D_{inner}^2)}{4} \times L \times \rho_{concrete}$$

**Conditions:**
- $F_{buoyancy} > 0$: Pipe floats (positions on tunnel crown)
- $F_{buoyancy} < 0$: Pipe sinks (positions on invert)
- $F_{buoyancy} \approx 0$: Neutral (unstable, avoid)

**Description:** Under groundwater conditions, determines if the pipe will float or sink, affecting line and grade control.

**Example:**
- Pipe: OD=2500mm, ID=2000mm, L=3m
- Concrete density = 2.42 t/m³
- Displaced water = π/4 × 2.5² × 3 × 1.0 = 14.73 t
- Pipe weight = π/4 × (2.5² - 2.0²) × 3 × 2.42 = 12.83 t
- Buoyancy = 14.73 - 12.83 = **1.90 t upward (FLOATS)**

---

### 4.6 Groundwater Force on Tunnel Face

#### Formula

$$F = A_{face} \times P \times 10$$

Where:
- $A_{face} = \frac{\pi \times D_{cutter}^2}{4}$ (m²)
- $P$ = water pressure at face (bar)
- $F$ = force (tonnes)

**Description:** Calculates the hydrostatic force the machine must resist. Critical for face support pressure and machine thrust calculations.

**Example:**
- Cutter = 2560mm → Area = 5.15 m²
- Water pressure = 0.6 bar
- Force = 5.15 × 0.6 × 10 = **30.9 tonnes**

**Note:** This force must be overcome by the slurry support pressure to prevent face collapse.

---

### 4.7 Specific Pipe Friction

#### Formula

$$R = \frac{F}{\pi \times D \times L}$$

Where:
- $R$ = specific friction (t/m²)
- $F$ = jacking force (tonnes)
- $D$ = pipe outer diameter (m)
- $L$ = pipe string length (m)

**Threshold Values:**
| Specific Friction | Status | Action |
|-------------------|--------|--------|
| < 0.5 t/m² | GOOD | Normal operation |
| 0.5 - 1.0 t/m² | ATTENTION | Monitor closely, check lubrication |
| > 1.0 t/m² | CRITICAL | Stop, investigate, remedial action |

**Description:** Normalized friction measurement that accounts for pipe size and length. Allows comparison across different projects and identification of problem sections.

**Example:**
- Force = 650 t
- Pipe = 2500mm OD, Length = 100m
- Surface area = π × 2.5 × 100 = 785 m²
- Specific friction = 650 / 785 = **0.83 t/m² (ATTENTION)**

---

### 4.8 Multi-Section Friction Analysis

#### Formula

$$R_i = \frac{F_i - F_{i-1}}{\pi \times D \times (L_i - L_{i-1})}$$

**Description:** When intermediate jacking stations (IJS) are used, friction can be calculated for individual sections to identify problem areas.

---

### 4.9 Advance Time with Cylinder Strokes

#### Formula

$$n_{strokes} = \lceil \frac{L_{pipe}}{L_{stroke}} \rceil$$

$$t_{pipe} = n_{strokes} \times \left(\frac{L_{stroke}}{v_{advance}} + \frac{L_{stroke}}{v_{retract}} + t_{changeover}\right) \times n_{sets}$$

Where:
- $n_{strokes}$ = strokes required per pipe
- $L_{stroke}$ = cylinder stroke length
- $v_{advance}$ = advance speed
- $v_{retract}$ = retract speed
- $t_{changeover}$ = time to switch cylinders
- $n_{sets}$ = number of cylinder sets (machine + expanders)

**Description:** Calculates total time to install one pipe, including all cylinder cycles and changeovers.

**Example:**
- Pipe = 3.0m, Stroke = 600mm → 5 strokes
- Advance = 10 cm/min, Retract = 20 cm/min
- Changeover = 1 min, 3 cylinder sets
- Time per stroke = (60/10) + (60/20) + 1 = 10 min
- Total = 5 × 10 + (extra cycles for sets) = **~70 min/pipe**

---

### 4.10 Settlement from Overcut

#### Method 1: Maximum Local Settlement

$$S_{max} = \frac{D_{cutter} - D_{pipe}}{2}$$

**Description:** Worst-case settlement if all overcut volume collapses directly above the pipe.

#### Method 2: Distributed Settlement

$$S_{avg} = \frac{V_{overcut/m}}{W_{influence}}$$

Where influence width ≈ tunnel diameter.

**Description:** More realistic average settlement assuming ground redistribution.

**Example:**
- Overcut = 30mm per side → $S_{max}$ = **30mm (3 cm)**
- Overcut volume = 0.238 m³/m, Width = 2.56m
- $S_{avg}$ = 0.238 / 2.56 = 0.093m = **9.3 cm**

**Note:** Actual settlement depends on ground conditions, overburden, and whether annular void is properly grouted.

---

### 4.11 Curve Radius from Joint Gap Measurements

#### Formula

$$\Delta_{gap} = |J_{left} - J_{right}|$$

$$\alpha = \arctan\left(\frac{\Delta_{gap}}{D_{pipe}}\right)$$

$$R = \frac{L_{pipe}}{\sin(\alpha)}$$

Where:
- $J_{left}$, $J_{right}$ = joint gaps at left and right springline
- $\alpha$ = pipe deflection angle
- $R$ = curve radius

**Joint Positions:**
- Peak (12 o'clock / crown)
- Right (3 o'clock / right springline)
- Invert (6 o'clock)
- Left (9 o'clock / left springline)

**Description:** Joint gap measurements indicate pipe deflection and can be used to verify curve radius during construction.

**Interpretation:**
| Gap Difference | Direction |
|----------------|-----------|
| Left > Right | Curving RIGHT |
| Right > Left | Curving LEFT |
| Peak > Invert | Bending DOWN |
| Invert > Peak | Bending UP |

---

### 4.12 Penetration Rate

#### Formula

$$P = \frac{v_{advance}}{RPM}$$

Where:
- $P$ = penetration rate (cm/revolution or mm/revolution)
- $v_{advance}$ = advance speed (cm/min)
- $RPM$ = cutterhead rotation speed

**Description:** Indicates how much the machine advances per rotation of the cutterhead. Important for understanding ground conditions and cutter wear.

**Typical Values:**
| Ground Type | Penetration Rate |
|-------------|------------------|
| Soft clay | 3-5 cm/rev |
| Stiff clay | 1-3 cm/rev |
| Sand | 2-4 cm/rev |
| Gravel | 1-2 cm/rev |
| Soft rock | 0.5-1.5 cm/rev |
| Hard rock | 0.1-0.5 cm/rev |

**Example:**
- Advance = 10 cm/min, RPM = 4.9
- Penetration = 10 / 4.9 = **2.04 cm/rev**

---

### 4.13 Slope/Grade Calculation

#### Formula

$$\Delta h = L \times \frac{slope\%}{100}$$

**Conversions:**
- Slope (%) = rise per 100m horizontal
- Slope (‰) = rise per 1000m = slope% × 10
- Slope ratio = 1 : (100 / slope%)

**Description:** Calculates height difference over a tunnel drive for vertical alignment planning.

**Example:**
- Distance = 465m, Slope = 1.2%
- Height difference = 465 × 0.012 = **5.58m**
- Slope ratio = 1 : 83

---

### 4.14 Ground Content from Slurry Density

#### Method 1: Single Measurement

$$G = (\rho_{slurry} - \rho_{base}) \times 1000$$

Where:
- $G$ = ground content (kg/m³)
- $\rho_{slurry}$ = measured slurry density (kg/L)
- $\rho_{base}$ = base fluid density (kg/L)

#### Method 2: Density Change

$$M_{ground} = (\rho_{final} - \rho_{initial}) \times V_{system} \times 1000$$

**Description:** Estimates excavated ground mass from slurry density measurements.

**Example:**
- Slurry = 1.10 kg/L, Base = 1.0 kg/L
- Ground content = (1.10 - 1.0) × 1000 = **100 kg/m³**

---

## Quick Reference Tables

### Soil Properties

| Soil Type | SG | Swell Factor | Friction Angle |
|-----------|-----|--------------|----------------|
| Clay | 2.55-2.70 | 1.20-1.35 | 0-15° |
| Silt | 2.65-2.70 | 1.15-1.25 | 25-30° |
| Sand | 2.60-2.70 | 1.10-1.20 | 30-38° |
| Gravel | 2.60-2.70 | 1.15-1.25 | 35-45° |
| Rock | 2.50-2.90 | 1.40-1.80 | - |

### Slurry Velocity Guidelines

| Condition | Minimum | Target | Maximum |
|-----------|---------|--------|---------|
| Fine sand | 2.5 m/s | 3.0 m/s | 4.0 m/s |
| Coarse sand | 3.0 m/s | 3.5 m/s | 4.5 m/s |
| Gravel | 3.5 m/s | 4.0 m/s | 5.0 m/s |
| Mixed | 3.0 m/s | 3.5 m/s | 4.5 m/s |

### Unit Conversions

| From | To | Multiply by |
|------|-----|-------------|
| bar | kPa | 100 |
| bar | psi | 14.5 |
| m³/hr | L/min | 16.67 |
| cm/min | m/hr | 0.6 |
| tonnes | kN | 9.81 |

### Concrete Pipe Weights

| Nominal Size | Wall Thickness | Weight/m |
|--------------|----------------|----------|
| DN800 | 100mm | ~0.7 t/m |
| DN1200 | 140mm | ~1.5 t/m |
| DN1600 | 160mm | ~2.5 t/m |
| DN2000 | 200mm | ~4.0 t/m |
| DN2400 | 250mm | ~6.5 t/m |

---

## References

1. **Excellence Pump Industry** - "Slurry Pump Manual: Fundamentals of Slurry Transport"

2. **Henke, L.** - "Hitchhiker's Guide to Pipejacking", Herrenknecht Academy Training Materials

3. **Durand, R. & Condolios, E.** (1952) - "Transport Hydraulique et Décantation des Matériaux Solides", Deuxièmes Journées de l'Hydraulique, Paris

4. **Wasp, E.J., Kenny, J.P., & Gandhi, R.L.** (1977) - "Solid-Liquid Flow Slurry Pipeline Transportation", Trans Tech Publications

5. **Duhme, R. et al.** - "Real-time Excavation Monitoring in Slurry Shield Tunneling", Tunnel Magazine

6. **German Pipe Jacking Association (GST)** - "Exercise Examples with Solutions" (Übungsbeispiele mit Lösungen)

7. **BS 5467 / BS 7211** - Cable current-carrying capacity tables

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | Jan 2026 | Initial release with all modules |

---

*Generated for MTBM Unified Calculator CLI*
