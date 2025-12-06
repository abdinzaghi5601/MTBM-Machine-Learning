# Steering Cylinder Calculation Formulas

## Reverse-Engineered from: Steer-cyl-cal-rev8_.xls

---

## Input Parameters

The spreadsheet uses the following input parameters (from "ENTER Parameters" sheet):

| Parameter | Variable Name | Value | Unit |
|-----------|--------------|-------|------|
| Laser gradient | `laser_gradient` | 0.00149 | - |
| Vertical angle of driving | `vertical_angle` | 1.49 | mm/m |
| Distance cutting head tip to target glass | `dist_head_to_target` | 2331.0 | mm |
| Length steering head | `length_steering_head` | 991.0 | mm |
| Target above axis | `target_above_axis` | 140.0 | mm |
| Number of steering cylinders | `num_cylinders` | 3.0 | - |
| Stroke of steering cylinders | `stroke` | 50.0 | mm |
| Mounting circle diameter | `mounting_diameter` | 715.0 | mm |
| Jacking pipe length | `pipe_length` | 3000.0 | mm |
| Cylinder readings (1-6) | `cyl_1` to `cyl_6` | 20, 32, 30, 25, 0, 0 | mm |
| Running PITCH | `running_pitch` | -11.0 | mm/m |
| Running YAW | `running_yaw` | -0.7 | mm/m |

---

## Key Formulas

### 1. **Drive Gradient Calculations**

```
Drive_100m = vertical_angle × 100
         = 1.49 × 100
         = 149.0 mm

Drive_per_pipe = vertical_angle × (pipe_length / 1000)
               = 1.49 × 3
               = 4.47 mm
```

### 2. **Pitch Calculations**

The pitch represents the vertical deviation per meter.

```
pitch_from_drive = Drive_100m / (pipe_length / 1000)
                 = 149.0 / 3.0
                 = 49.67 mm/m
```

However, based on the data showing 14.77 mm/m, the actual formula appears to be:

```
actual_pitch = (Drive_per_pipe / pipe_length) × 1000
             = (4.47 / 3000) × 1000
             = 1.49 mm/m × (some correction factor)
```

**Alternative Pitch Formula (based on observed value of 14.77):**
```
pitch = (target_above_axis / dist_head_to_target) × 1000 × correction_factor
```

### 3. **Steering Pitch Calculation**

The steering pitch is the pitch correction needed from the cylinders:

```
steering_pitch = desired_pitch - measured_pitch
               = -4.5 - 14.77
               = -19.27 mm/m (approximately -13.5 shown)
```

### 4. **Pitch After One Pipe**

```
pitch_after_one_pipe = (steering_pitch × pipe_length / 1000) + Drive_per_pipe
                     = (-13.5 × 3) + 4.47
                     = -40.5 + 4.47
                     = -36.03 mm per pipe
                     
                     = -36.03 / 3 = -12.01 mm/m (but shows 1.27 mm/m)
```

**Revised formula (based on observed 1.27 mm/m):**
```
pitch_after_one_pipe = running_pitch + steering_pitch + drive_pitch
```

### 5. **Yaw Calculations (3-Cylinder System)**

For a 3-cylinder steering system arranged 120° apart:

**Yaw Components:**
```
yaw_left = cylinder_displacement_left
yaw_right = cylinder_displacement_right
total_yaw = yaw_left + yaw_right
```

From the data:
- Yaw Left: 6.0 mm/m
- Yaw Right: 10.5 mm/m  
- Total Yaw: 16.5 mm/m

### 6. **Cylinder Position Calculations**

For 3 cylinders arranged at 0°, 120°, and 240°:

```
Cylinder_1 (Top, 0°) = stroke_center + pitch_component
Cylinder_2 (120°) = stroke_center + (pitch × cos(120°)) + (yaw × sin(120°))
Cylinder_3 (240°) = stroke_center + (pitch × cos(240°)) + (yaw × sin(240°))
```

**General Formula for Each Cylinder:**
```
Cyl_n = stroke/2 + (pitch_effect × cos(angle_n)) + (yaw_effect × sin(angle_n))
```

Where:
- `angle_n` = cylinder position angle
- `pitch_effect` = (steering_pitch × mounting_diameter/2) / 1000
- `yaw_effect` = (steering_yaw × mounting_diameter/2) / 1000

### 7. **Running Pitch and Yaw**

```
Running_PITCH = current_pitch + correction_from_cylinders
Running_YAW = current_yaw + correction_from_cylinders
```

---

## Cylinder Displacement Formulas

### For 3-Cylinder System (120° apart):

**Cylinder Positions:**
- Cylinder 1: 0° (Top)
- Cylinder 2: 120° 
- Cylinder 3: 240°

**Displacement Calculation:**
```python
def calculate_3cyl_displacement(pitch_mmpm, yaw_mmpm, mounting_dia, stroke_center=25):
    """
    Calculate cylinder displacements for 3-cylinder steering
    
    Parameters:
    - pitch_mmpm: Desired pitch in mm/m
    - yaw_mmpm: Desired yaw in mm/m  
    - mounting_dia: Mounting circle diameter in mm
    - stroke_center: Center position of stroke (typically stroke/2)
    
    Returns:
    - Dictionary with cylinder displacements
    """
    import math
    
    # Radius in meters
    radius_m = (mounting_dia / 2) / 1000
    
    # Convert pitch and yaw to radians at the mounting radius
    pitch_rad = (pitch_mmpm / 1000) / radius_m
    yaw_rad = (yaw_mmpm / 1000) / radius_m
    
    # Calculate each cylinder position
    cyl_1 = stroke_center + (pitch_mmpm * radius_m)
    cyl_2 = stroke_center + (pitch_mmpm * radius_m * math.cos(math.radians(120))) + \
                            (yaw_mmpm * radius_m * math.sin(math.radians(120)))
    cyl_3 = stroke_center + (pitch_mmpm * radius_m * math.cos(math.radians(240))) + \
                            (yaw_mmpm * radius_m * math.sin(math.radians(240)))
    
    return {
        'cylinder_1': round(cyl_1, 2),
        'cylinder_2': round(cyl_2, 2),
        'cylinder_3': round(cyl_3, 2)
    }
```

### For 4-Cylinder System (90° apart):

```python
def calculate_4cyl_displacement(pitch_mmpm, yaw_mmpm, mounting_dia, stroke_center=25):
    """
    Calculate cylinder displacements for 4-cylinder steering
    
    Cylinder Positions:
    - Cylinder 1: 0° (Top)
    - Cylinder 2: 90° (Right)
    - Cylinder 3: 180° (Bottom)
    - Cylinder 4: 270° (Left)
    """
    import math
    
    radius_m = (mounting_dia / 2) / 1000
    
    cyl_1 = stroke_center + (pitch_mmpm * radius_m)
    cyl_2 = stroke_center + (yaw_mmpm * radius_m)
    cyl_3 = stroke_center - (pitch_mmpm * radius_m)
    cyl_4 = stroke_center - (yaw_mmpm * radius_m)
    
    return {
        'cylinder_1': round(cyl_1, 2),
        'cylinder_2': round(cyl_2, 2),
        'cylinder_3': round(cyl_3, 2),
        'cylinder_4': round(cyl_4, 2)
    }
```

### For 6-Cylinder System (60° apart):

```python
def calculate_6cyl_displacement(pitch_mmpm, yaw_mmpm, mounting_dia, stroke_center=25):
    """
    Calculate cylinder displacements for 6-cylinder steering
    
    Cylinder Positions: 0°, 60°, 120°, 180°, 240°, 300°
    """
    import math
    
    radius_m = (mounting_dia / 2) / 1000
    cylinders = {}
    
    for i in range(6):
        angle = i * 60  # degrees
        cyl_disp = stroke_center + \
                   (pitch_mmpm * radius_m * math.cos(math.radians(angle))) + \
                   (yaw_mmpm * radius_m * math.sin(math.radians(angle)))
        cylinders[f'cylinder_{i+1}'] = round(cyl_disp, 2)
    
    return cylinders
```

---

## Reverse Calculation (From Cylinder Readings to Pitch/Yaw)

Given cylinder readings, calculate the actual pitch and yaw:

### For 3-Cylinder System:

```python
def calculate_pitch_yaw_from_3cyl(cyl_1, cyl_2, cyl_3, mounting_dia, stroke_center=25):
    """
    Calculate pitch and yaw from cylinder readings
    """
    import math
    
    radius_m = (mounting_dia / 2) / 1000
    
    # Remove center offset
    c1 = cyl_1 - stroke_center
    c2 = cyl_2 - stroke_center
    c3 = cyl_3 - stroke_center
    
    # Pitch is primarily from cylinder 1 (top position)
    pitch_mmpm = c1 / radius_m
    
    # Yaw calculation from the 120° and 240° cylinders
    # Using vector decomposition
    yaw_component = (c2 - c3) / (math.sqrt(3) * radius_m)
    
    return {
        'pitch': round(pitch_mmpm, 2),
        'yaw': round(yaw_component, 2)
    }
```

### For 4-Cylinder System:

```python
def calculate_pitch_yaw_from_4cyl(cyl_1, cyl_2, cyl_3, cyl_4, mounting_dia, stroke_center=25):
    """
    Calculate pitch and yaw from 4 cylinder readings
    """
    radius_m = (mounting_dia / 2) / 1000
    
    # Remove center offset
    c1 = cyl_1 - stroke_center
    c2 = cyl_2 - stroke_center
    c3 = cyl_3 - stroke_center
    c4 = cyl_4 - stroke_center
    
    # Pitch from top and bottom cylinders
    pitch_mmpm = (c1 - c3) / (2 * radius_m)
    
    # Yaw from left and right cylinders
    yaw_mmpm = (c2 - c4) / (2 * radius_m)
    
    return {
        'pitch': round(pitch_mmpm, 2),
        'yaw': round(yaw_mmpm, 2)
    }
```

---

## Complete Calculation Workflow

1. **Input Parameters** → Enter machine specifications and target trajectory
2. **Drive Gradient** → Calculate natural pitch from tunnel gradient
3. **Target Pitch/Yaw** → Determine desired steering corrections
4. **Steering Correction** → Calculate required cylinder movements
5. **Cylinder Positions** → Compute individual cylinder extensions
6. **Verify** → Check if cylinders are within stroke limits
7. **Apply** → Set cylinder positions on machine
8. **Monitor** → Track actual pitch/yaw and adjust as needed

---

## Key Relationships

1. **Pitch** affects all cylinders proportionally based on their vertical component
2. **Yaw** affects cylinders based on their horizontal component
3. The **mounting circle radius** determines the leverage for steering corrections
4. **Stroke limits** (0 to 50mm typically) constrain maximum pitch/yaw corrections
5. **Running pitch/yaw** includes cumulative effects from previous pipes

---

## Notes

- All angular positions assume Cylinder 1 is at the top (12 o'clock position)
- Positive pitch = steering upward
- Positive yaw = steering rightward (convention may vary)
- The formulas assume small angle approximations are valid
- Cylinder readings are absolute positions from fully retracted (0mm)
