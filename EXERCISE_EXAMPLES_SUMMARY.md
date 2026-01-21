# Technical Mathematics Exercise Examples - Summary

**Source**: exercise_examples_with_solutions_GB.pdf  
**Pages**: 6  
**Type**: Exercise examples with complete solutions

---

## Overview

This document contains practical calculation exercises for microtunneling operations with complete step-by-step solutions. It covers various aspects of MTBM operations including volumes, flow rates, forces, pressures, and advance calculations.

---

## Exercise Topics Covered

### **1. Overcut Volume Calculations (Exercise a)**
- Calculate theoretical overcut volume per running meter
- Formula: V_Ü = π/4 × (D_a² - d_a²) × 1.0 m
- Example: DN 2000, Advance pipe D_a = 2500 mm, Cutterhead D_a = 2560 mm
- Result: 0.238 m³ per meter

### **2. Bentonite Suspension Flow Rate (Exercise b)**
- Calculate required bentonite flow to fill overcut volume
- Given: Advance speed = 10 cm/min
- Formula: V̇ = V_Ü × v
- Result: 23.8 l/min

### **3. Total Volume Flow with Losses (Exercise c)**
- Account for ground/environment losses (15%)
- Formula: V_total = V_base × (1 + loss_percentage)
- Result: 27.4 l/min

### **4. Lubrication Stations (Exercise d)**
- Calculate number of lubrication stations needed
- Given: Advance distance 325 m, machine length 10 m, 5 standard pipes (3 m each)
- Calculate stations between lubrication pipes
- Result: 21 stations, 1.05 m³/pipe

### **5. Bentonite Pump Requirements (Exercise e)**
- Calculate total bentonite volume needed
- Includes machine section and pipe section
- Formula: V_Bentonite = 0.8 × (V_Machine + V_Pipe)
- Result: 554.4 kg/day

### **6. Ground Excavation Volume (Exercise g)**
- Calculate volume excavated per advance pipe
- Formula: V_Pipe = π/4 × D_Ü² × L_Pipe
- Example: 15.44 m³ per pipe

### **7. Swell Factor and Separation (Exercise h)**
- Account for swell factor (1.2) and separation efficiency (95%)
- Formula: V_Excavation = 1.2 × 0.95 × V_Pipe × D_Ü²/4 × π
- Result: 17.60 m³

### **8. Ground Mass Calculation (Exercise i)**
- Calculate mass of excavated material
- Given: Material density = 1.9 t/m³
- Formula: M_Ground = ρ × V_Excavation
- Result: 33.44 t

### **9. Bentonite Volume for Sedimentation Tank (Exercise j)**
- Calculate bentonite needed for tank renewal
- Given: Tank volume 50 m³, 30 kg bentonite per m³
- Result: 1,500 kg

### **10. Buoyancy Calculation (Exercise k)**
- Determine if pipes are subject to buoyancy
- Calculate buoyancy force = Mass of water displaced - Mass of pipe
- Given: Ground water 5 m above pipe peak, ρ_concrete = 2.42 t/m³
- Result: Each pipe subject to 1.8 t buoyancy force

### **11. Ground Water Pressure on Machine (Exercise l)**
- Calculate force from ground water pressure
- Given: Average pressure 0.6 bar at tunnel face
- Formula: F_GW = P × A = P × π/4 × D_Ü²
- Result: 30.88 t (31 t) - Machine should be secured!

### **12. Pipe Friction Calculations (Exercise m)**
- Calculate specific pipe friction (t/m²) for different sections
- Given: Advance status 325 m, 2 expanders installed
- Sections:
  - Section 1 (Expander 1 - Machine): 0.64 t/m²
  - Section 2 (Expander 2 - Expander 1): 0.83 t/m²
  - Section 3 (Main cylinder - Expander 2): 0.63 t/m²
- **Guidelines**:
  - < 0.5 t/m²: Positive values ✅
  - 0.5 - 1.0 t/m²: Require special attention ⚠️
  - > 1.0 t/m²: Critical values ❌

### **13. Advance Time Calculation (Exercise n)**
- Calculate time required for one pipe advance
- Given: Expander stroke 600 mm, change-over time 1 min, speeds vary
- Components:
  - Expander 1 to machine: 10 cm/min
  - Expander 2: 10 cm/min
  - Main cylinder: 20 cm/min
- Result: 70 minutes per pipe

### **14. Thrust Force on Expander (Exercise o)**
- Calculate thrust force on expander 2
- Given: 16 cylinders (Ø 140 mm) installed
- (Calculation details in document)

### **15. Height Difference Calculation (Exercise p)**
- Calculate height difference between shafts
- Given: Distance 465 m, upward slope 1.2%
- Formula: ΔH = L × slope%
- Result: 5.58 m

### **16. Ground Excavation from Flow Rates (Exercise q)**
- Calculate theoretically excavated ground from flow rate difference
- Given: Feed line 180 m³/h, Slurry line 213 m³/h, Time 15 min
- Formula: V_B = (V_slurry - V_feed) × time
- Result: 8.25 m³ in 15 minutes

### **17. Cutterhead Penetration (Exercise r)**
- Calculate penetration per rotation
- Given: Advance speed 10 cm/min, Cutterhead 4.9 rpm
- Formula: P = V / U (rotations)
- Result: 2.04 cm/rotation
- **Note**: If speed reduced to 2 rpm, penetration increases to 5 cm/rotation

### **18. Ground Material in Advance Water (Exercise s)**
- Calculate ground material content in advance water
- Given: Density of advance water 1.10 kg/l
- Pure water density: 1.0 kg/l
- Result: 0.1 kg/l ground material

### **19. Ground Entry into Advance Water (Exercise t)**
- Calculate how much ground entered advance water during pipe advance
- Given: Initial density 1.05 kg/l, Final density 1.08 kg/l, Volume 50 m³
- (Calculation details in document)

---

## Key Formulas Used

### **Volume Calculations**:
- Overcut volume: V = π/4 × (D² - d²) × L
- Cylindrical volume: V = π/4 × D² × L

### **Flow Rate Calculations**:
- Flow rate: Q = V × v (volume × velocity)
- With losses: Q_total = Q_base × (1 + loss%)

### **Force Calculations**:
- Force from pressure: F = P × A
- Buoyancy: F_A = M_water - M_pipe

### **Friction Calculations**:
- Specific friction: R = F / (L × U × A)
- Where: L = length, U = circumference, A = area

### **Time Calculations**:
- Time = Distance / Speed
- Total time = Sum of all operation times

---

## Important Guidelines from Document

### **Pipe Friction Values**:
- **< 0.5 t/m²**: Positive values ✅
- **0.5 - 1.0 t/m²**: Require special attention ⚠️
- **> 1.0 t/m²**: Critical values ❌

### **Safety Considerations**:
- Ground water pressure can exert significant force (31 t in example)
- Machine should be secured when starting advance
- Buoyancy forces must be considered below ground water level

---

## Practical Applications

These exercises demonstrate calculations for:
1. **System Design**: Sizing pumps, tanks, and flow systems
2. **Operation Planning**: Advance times, lubrication requirements
3. **Safety Analysis**: Buoyancy, ground water pressure, friction
4. **Material Management**: Excavation volumes, bentonite requirements
5. **Performance Monitoring**: Penetration rates, flow rates, densities

---

## Files Created

1. **exercise_examples_with_solutions_extracted.txt** - Full extracted text
2. **read_exercise_examples.py** - Extraction script
3. **EXERCISE_EXAMPLES_SUMMARY.md** - This summary

---

## Next Steps

You can now:
1. Reference specific exercises by letter (a, b, c, etc.)
2. Use formulas for your own calculations
3. Create calculators based on these exercises
4. Integrate with your existing systems

Would you like me to create calculators for any of these specific exercises?



