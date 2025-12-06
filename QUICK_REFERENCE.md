# STEERING CYLINDER QUICK REFERENCE

**Supported Systems:** 3-Cylinder, 4-Cylinder, and 6-Cylinder

## FORMULAS AT A GLANCE

### 3-Cylinder System (120° Spacing)

#### Forward: Pitch/Yaw → Cylinders
```
R = mounting_diameter / 2000  (meters)
C = stroke / 2  (center position)

Cylinder 1 (0°)   = C + (Pitch × R)
Cylinder 2 (120°) = C + (Pitch × R × -0.5) + (Yaw × R × 0.866)
Cylinder 3 (240°) = C + (Pitch × R × -0.5) - (Yaw × R × 0.866)
```

#### Reverse: Cylinders → Pitch/Yaw
```
c1 = Cyl1 - C
c2 = Cyl2 - C  
c3 = Cyl3 - C

Pitch = c1 / R
Yaw   = (c2 - c3) / (1.732 × R)
```

### 4-Cylinder System (90° Spacing)

#### Forward: Pitch/Yaw → Cylinders
```
Cylinder 1 (0°)   = C + (Pitch × R)
Cylinder 2 (90°)  = C + (Yaw × R)
Cylinder 3 (180°) = C - (Pitch × R)
Cylinder 4 (270°) = C - (Yaw × R)
```

#### Reverse: Cylinders → Pitch/Yaw
```
Pitch = (Cyl1 - Cyl3) / (2 × R)
Yaw   = (Cyl2 - Cyl4) / (2 × R)
```

### 6-Cylinder System (60° Spacing)

#### Forward: Pitch/Yaw → Cylinders
```
For each cylinder i (1 to 6):
  angle = (i-1) × 60°
  Cylinder[i] = C + (Pitch × R × cos(angle)) + (Yaw × R × sin(angle))
```

#### Reverse: Cylinders → Pitch/Yaw
```
Pitch = (Cyl1 - Cyl4) / (2 × R)
Yaw   = ((Cyl2 - Cyl5) + (Cyl3 - Cyl6)) / (2 × √3 × R)
```

**Cylinder Arrangement:**
```
Cyl 1:   0° (Top)
Cyl 2:  60°
Cyl 3: 120°
Cyl 4: 180° (Bottom)
Cyl 5: 240°
Cyl 6: 300°
```

---

## TYPICAL VALUES

| Parameter | Typical Range | Your Value |
|-----------|--------------|------------|
| Mounting Diameter | 600-900 mm | 715 mm |
| Stroke | 40-60 mm | 50 mm |
| Pipe Length | 2500-3500 mm | 3000 mm |
| Max Pitch | ±50 mm/m | - |
| Max Yaw | ±50 mm/m | - |

---

## SIGN CONVENTIONS

```
         Pitch+
            ↑
            |
Yaw- ←─────┼─────→ Yaw+
            |
            ↓
         Pitch-
```

- **Positive Pitch** = Steering UP
- **Negative Pitch** = Steering DOWN  
- **Positive Yaw** = Steering RIGHT
- **Negative Yaw** = Steering LEFT

---

## CYLINDER POSITIONS (3-CYL)

```
        Cyl 1 (0°)
           ┃
           ┃
    ╔═════╬═════╗
    ║     ┃     ║
Cyl3║     ●     ║Cyl2
240°║           ║120°
    ║           ║
    ╚═══════════╝
```

---

## QUICK CHECKS

### ✓ Cylinder Limits
- Minimum: 0 mm
- Maximum: 50 mm
- Safe zone: 5-45 mm
- Warning: <5mm or >45mm

### ✓ Steering Limits
- Normal pitch: ±20 mm/m
- Normal yaw: ±20 mm/m
- Max recommended: ±40 mm/m
- Extreme: ±50 mm/m

### ✓ Correction Per Pipe
```
Correction = (Pitch or Yaw) × (Pipe_Length / 1000)

Example: 
  Pitch = -10 mm/m
  Pipe = 3000 mm
  Correction = -10 × 3 = -30 mm per pipe
```

---

## COMMON SCENARIOS

### Scenario 1: Steering UP
```
Target: Pitch = +10 mm/m, Yaw = 0
Result: Cyl1 HIGH, Cyl2/3 LOW
```

### Scenario 2: Steering DOWN
```
Target: Pitch = -10 mm/m, Yaw = 0
Result: Cyl1 LOW, Cyl2/3 HIGH
```

### Scenario 3: Steering RIGHT
```
Target: Pitch = 0, Yaw = +10 mm/m
Result: Cyl2 HIGH, Cyl3 LOW, Cyl1 MID
```

### Scenario 4: Steering LEFT
```
Target: Pitch = 0, Yaw = -10 mm/m
Result: Cyl2 LOW, Cyl3 HIGH, Cyl1 MID
```

---

## FIELD PROCEDURE

1. **Measure Current Position**
   - Record all cylinder readings
   - Calculate current pitch/yaw

2. **Determine Target**
   - Check laser alignment
   - Calculate desired pitch/yaw

3. **Calculate Correction**
   - Target - Current = Required correction
   - Check if within limits

4. **Compute New Positions**
   - Apply formulas
   - Verify stroke limits

5. **Apply & Monitor**
   - Set cylinder positions
   - Advance one pipe
   - Re-measure and adjust

---

## EXAMPLE CALCULATION

### Given:
- Current: Pitch = +5 mm/m, Yaw = -3 mm/m
- Target: Pitch = -2 mm/m, Yaw = +5 mm/m
- Mounting Ø: 715 mm → R = 0.3575 m

### Calculate:
```
Required correction:
  Pitch: -2 - (+5) = -7 mm/m
  Yaw: +5 - (-3) = +8 mm/m

Cylinder positions:
  C1 = 25 + (-7 × 0.3575) = 22.50 mm
  C2 = 25 + (-7 × 0.3575 × -0.5) + (8 × 0.3575 × 0.866) = 28.73 mm
  C3 = 25 + (-7 × 0.3575 × -0.5) - (8 × 0.3575 × 0.866) = 21.27 mm

All within limits ✓
```

---

## TROUBLESHOOTING

| Problem | Cause | Solution |
|---------|-------|----------|
| Cylinder >50mm | Too much correction | Reduce pitch/yaw |
| Cylinder <0mm | Too much correction | Reduce pitch/yaw |
| No effect | Wrong sign | Check +/- convention |
| Opposite effect | Wrong cylinder | Verify cylinder numbering |
| Gradual drift | Gradient not accounted | Add drive pitch component |

---

## PYTHON ONE-LINERS

```python
# Calculate cylinders from pitch/yaw
from steering_calculator import *
p = MachineParameters(mounting_diameter=715, stroke=50)
calc = SteeringCalculator(p)
result = calc.calculate_3cyl_displacement(SteeringCommand(pitch=-5, yaw=10))
print(result)

# Calculate pitch/yaw from cylinders
result = calc.calculate_pitch_yaw_from_3cyl(20, 32, 30)
print(f"Pitch: {result.pitch}, Yaw: {result.yaw}")
```

---

## EMERGENCY CONTACTS

- Machine Operator: _____________
- Site Engineer: _____________
- Technical Support: _____________

## NOTES

```
_________________________________________________

_________________________________________________

_________________________________________________

_________________________________________________
```

---

**Document Version:** 1.0  
**Last Updated:** 2024-12-04  
**Reference:** Steer-cyl-cal-rev8_.xls
