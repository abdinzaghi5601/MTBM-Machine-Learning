# Steering Formulas Explained: How to Correct Deviations to Zero

## Understanding the Goal

**Objective**: Bring the machine's **line** (horizontal deviation) and **level** (vertical deviation) to **0 mm** by applying progressive steering corrections over multiple pipe installations.

---

## Core Concepts

### 1. **Pitch and Yaw - What They Mean**

- **Pitch (mm/m)**: Vertical steering rate
  - Positive pitch = steering upward
  - Negative pitch = steering downward
  - **Target: 0 mm/m** (level)

- **Yaw (mm/m)**: Horizontal steering rate  
  - Positive yaw = steering right
  - Negative yaw = steering left
  - **Target: 0 mm/m** (straight line)

### 2. **How Cylinders Control Steering**

The steering cylinders are arranged in a circle around the machine. By extending/retracting different cylinders, you tilt the steering head, which changes the direction the machine travels.

**Key Formula**:
```
Cylinder Position = Center Position + (Pitch Effect × cos(angle)) + (Yaw Effect × sin(angle))
```

Where:
- **Center Position** = stroke/2 (typically 25mm for 50mm stroke)
- **Pitch Effect** = (pitch_mm_per_m × mounting_radius_meters)
- **Yaw Effect** = (yaw_mm_per_m × mounting_radius_meters)
- **angle** = cylinder position angle (0°, 120°, 240° for 3-cylinder system)

---

## The Correction Process: Step-by-Step

### **Step 1: Measure Current Deviation**

From your MTBM data, you get:
- Current horizontal deviation: `hor_deviation_machine_mm`
- Current vertical deviation: `vert_deviation_machine_mm`
- Current pitch: `pitch_mm_per_m`
- Current yaw: `yaw_mm_per_m`

### **Step 2: Calculate Required Correction**

To bring deviations to zero, you need to:

```
Required Pitch Correction = 0 - current_pitch = -current_pitch
Required Yaw Correction = 0 - current_yaw = -current_yaw
```

**Example**:
- Current pitch: +10.3 mm/m (steering upward)
- Current yaw: -9.7 mm/m (steering left)
- **Required correction**: Pitch = -10.3 mm/m, Yaw = +9.7 mm/m

### **Step 3: Calculate Cylinder Positions**

For a **3-cylinder system** (120° spacing):

```
Cylinder 1 (Top, 0°):
  = 25 + (pitch_correction × radius_m)

Cylinder 2 (120°):
  = 25 + (pitch_correction × radius_m × cos(120°)) + (yaw_correction × radius_m × sin(120°))

Cylinder 3 (240°):
  = 25 + (pitch_correction × radius_m × cos(240°)) + (yaw_correction × radius_m × sin(240°))
```

Where `radius_m = (mounting_diameter / 2) / 1000`

### **Step 4: Apply Correction Over Time**

**Critical Understanding**: You cannot correct everything in one pipe! Corrections must be applied progressively.

**Why?**
- Each pipe is only 3 meters long
- Maximum cylinder stroke limits how much correction you can apply
- Aggressive corrections can cause instability

**Correction Per Pipe**:
```
Correction per pipe = steering_rate × (pipe_length / 1000)
```

**Example**:
- Pipe length: 3000mm = 3 meters
- Pitch correction: -10.3 mm/m
- **Correction after one pipe**: -10.3 × 3 = -30.9 mm

### **Step 5: Monitor and Adjust**

After installing one pipe:
1. Measure new pitch/yaw
2. Calculate remaining deviation
3. Plan next correction
4. Repeat until deviations are near zero

---

## Complete Formula Reference

### **Forward Calculation: Pitch/Yaw → Cylinders**

#### For 3-Cylinder System:

```python
radius_m = (mounting_diameter / 2) / 1000  # Convert to meters
stroke_center = stroke / 2  # Typically 25mm

pitch_effect = pitch_mm_per_m × radius_m
yaw_effect = yaw_mm_per_m × radius_m

cylinder_1 = stroke_center + pitch_effect
cylinder_2 = stroke_center + (pitch_effect × cos(120°)) + (yaw_effect × sin(120°))
cylinder_3 = stroke_center + (pitch_effect × cos(240°)) + (yaw_effect × sin(240°))
```

#### For 4-Cylinder System:

```python
cylinder_1 = stroke_center + pitch_effect  # Top
cylinder_2 = stroke_center + yaw_effect     # Right
cylinder_3 = stroke_center - pitch_effect  # Bottom
cylinder_4 = stroke_center - yaw_effect    # Left
```

### **Reverse Calculation: Cylinders → Pitch/Yaw**

#### For 3-Cylinder System:

```python
c1_offset = cylinder_1 - stroke_center
c2_offset = cylinder_2 - stroke_center
c3_offset = cylinder_3 - stroke_center

pitch = c1_offset / radius_m
yaw = (c2_offset - c3_offset) / (√3 × radius_m)
```

#### For 4-Cylinder System:

```python
pitch = (cylinder_1 - cylinder_3) / (2 × radius_m)
yaw = (cylinder_2 - cylinder_4) / (2 × radius_m)
```

---

## Progressive Correction Strategy

### **Scenario: Machine is 50mm off horizontally and 30mm off vertically**

**Initial State**:
- Horizontal deviation: 50mm
- Vertical deviation: 30mm
- Current pitch: +10 mm/m
- Current yaw: -15 mm/m

**Goal**: Bring both to 0mm

### **Correction Plan (Multiple Pipes)**

#### **Pipe 1: Initial Correction**
```
Target: Reduce deviations by 50%
Required pitch correction: -5 mm/m (half of current)
Required yaw correction: +7.5 mm/m (half of current)

Expected after Pipe 1:
- New pitch: +5 mm/m
- New yaw: -7.5 mm/m
- Horizontal deviation: ~25mm (reduced)
- Vertical deviation: ~15mm (reduced)
```

#### **Pipe 2: Further Correction**
```
Target: Reduce remaining deviations by 50%
Required pitch correction: -2.5 mm/m
Required yaw correction: +3.75 mm/m

Expected after Pipe 2:
- New pitch: +2.5 mm/m
- New yaw: -3.75 mm/m
- Horizontal deviation: ~12mm
- Vertical deviation: ~7mm
```

#### **Pipe 3: Fine Correction**
```
Target: Bring to near zero
Required pitch correction: -2.5 mm/m
Required yaw correction: -3.75 mm/m

Expected after Pipe 3:
- New pitch: 0 mm/m
- New yaw: 0 mm/m
- Horizontal deviation: ~0mm
- Vertical deviation: ~0mm
```

---

## Ground Condition Constraints ⚠️ CRITICAL

**Ground condition is CRITICAL for steering corrections!** Different ground types have different limits to prevent jacking pressure increase and procedure halting.

### **Soft Geology**
- **Maximum steering rate**: 10 mm/m
- **Recommended max**: 8 mm/m
- Can handle more aggressive steering corrections
- Less risk of jacking pressure increase

### **Mixed Geology**
- **Maximum steering rate**: 4 mm/m
- **Recommended max**: 3 mm/m
- **CRITICAL**: Limit to 2-4 mm/m to avoid jacking pressure increase
- Exceeding limits can cause significant pressure rise

### **Rock Geology**
- **Maximum steering rate**: 2 mm/m
- **Recommended max**: 1.5 mm/m
- **EXTREMELY CRITICAL**: More than 2 mm/m can cause:
  - Jacking pressure increase
  - Procedure halting
  - Equipment damage risk
- Must take corrections with utmost care
- Monitor jacking pressure closely

### **How Ground Condition Affects Corrections**

The system automatically limits corrections based on ground condition:

```
If requested_correction > max_allowed_for_ground:
    correction = scale_down_to_max_allowed
    Warning: "Correction limited due to ground condition"
```

**Example**:
- Requested: Pitch = -8 mm/m, Yaw = +6 mm/m (total = 10 mm/m)
- Ground: Rock (max 2 mm/m)
- **Result**: Automatically scaled to Pitch = -1.6 mm/m, Yaw = +1.2 mm/m (total = 2 mm/m)

## Key Principles for Successful Correction

### 1. **Consider Ground Condition FIRST** ⚠️
- Always specify ground condition before planning corrections
- System will automatically limit corrections to safe levels
- Rock ground requires very careful, gradual corrections

### 2. **Don't Over-Correct**
- Large corrections can overshoot the target
- Better to correct gradually over 2-3 pipes
- Especially important in rock ground

### 3. **Monitor Continuously**
- Check pitch/yaw after each pipe
- Monitor jacking pressure (especially in rock)
- Adjust plan based on actual results

### 4. **Account for Natural Drift**
- Machines naturally drift due to ground conditions
- May need small corrections even when "on target"

### 5. **Stay Within Limits**
- Cylinder positions must be between 0 and stroke_max
- Ground condition limits must be respected
- If correction requires out-of-range cylinders, split into smaller steps

### 6. **Consider Gradient Effects**
- Tunnel gradient affects natural pitch
- May need to compensate for gradient in corrections

---

## Practical Example: Complete Correction Sequence

### **Starting Conditions**:
```
Current Pitch: +12.5 mm/m (steering up)
Current Yaw: -18.3 mm/m (steering left)
Horizontal Deviation: 60mm
Vertical Deviation: 40mm
```

### **Correction Sequence**:

#### **Pipe 1**:
```
Required Correction: Pitch = -6.25 mm/m, Yaw = +9.15 mm/m
Cylinder Positions: [Calculate using formulas]
Expected Result: Pitch = +6.25 mm/m, Yaw = -9.15 mm/m
```

#### **Pipe 2**:
```
Required Correction: Pitch = -3.12 mm/m, Yaw = +4.58 mm/m
Expected Result: Pitch = +3.12 mm/m, Yaw = -4.58 mm/m
```

#### **Pipe 3**:
```
Required Correction: Pitch = -3.12 mm/m, Yaw = -4.58 mm/m
Expected Result: Pitch = 0 mm/m, Yaw = 0 mm/m
```

#### **Pipe 4** (Fine-tuning):
```
Required Correction: Pitch = 0 mm/m, Yaw = 0 mm/m
Maintain alignment
```

---

## Formula Summary

### **Core Relationships**:

1. **Cylinder Position**:
   ```
   Cyl_n = center + (pitch × radius × cos(θ_n)) + (yaw × radius × sin(θ_n))
   ```

2. **Correction Per Pipe**:
   ```
   Correction_mm = steering_rate_mm_per_m × pipe_length_meters
   ```

3. **Progressive Correction**:
   ```
   Next_pitch = current_pitch + pitch_correction
   Next_yaw = current_yaw + yaw_correction
   ```

4. **Target Achievement**:
   ```
   Required_correction = target - current
   ```

---

## Important Notes

1. **Units Matter**: Always convert consistently (mm, meters, mm/m)

2. **Sign Conventions**: 
   - Positive pitch = up
   - Positive yaw = right (verify for your system)

3. **Limits**: Cylinders typically range 0-50mm stroke

4. **Time Factor**: Corrections take effect over the length of one pipe (typically 3m)

5. **Measurement**: Always verify actual pitch/yaw after each pipe installation

---

## Next Steps

Use the `steering_correction_simulator.py` script to see these formulas in action with real examples!

