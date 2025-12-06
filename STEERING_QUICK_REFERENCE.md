# Steering Correction Quick Reference

## Goal: Bring Line & Level to Zero

**Line (Horizontal) = 0 mm**  
**Level (Vertical) = 0 mm**

This means: **Pitch = 0 mm/m** and **Yaw = 0 mm/m**

---

## The Formula (3-Cylinder System)

### Forward: Pitch/Yaw → Cylinder Positions

```
radius_m = (mounting_diameter / 2) / 1000
stroke_center = stroke / 2  (typically 25mm)

pitch_effect = pitch_mm_per_m × radius_m
yaw_effect = yaw_mm_per_m × radius_m

Cylinder 1 (Top, 0°):
  = stroke_center + pitch_effect

Cylinder 2 (120°):
  = stroke_center + (pitch_effect × -0.5) + (yaw_effect × 0.866)

Cylinder 3 (240°):
  = stroke_center + (pitch_effect × -0.5) + (yaw_effect × -0.866)
```

### Reverse: Cylinder Positions → Pitch/Yaw

```
c1_offset = cylinder_1 - stroke_center
c2_offset = cylinder_2 - stroke_center
c3_offset = cylinder_3 - stroke_center

pitch = c1_offset / radius_m
yaw = (c2_offset - c3_offset) / (1.732 × radius_m)
```

---

## Correction Process

### Step 1: Measure Current State
- Current Pitch: `current_pitch` mm/m
- Current Yaw: `current_yaw` mm/m

### Step 2: Calculate Required Correction
```
pitch_correction = 0 - current_pitch = -current_pitch
yaw_correction = 0 - current_yaw = -current_yaw
```

### Step 3: Apply Progressively (Don't Over-Correct!)

**Strategy**: Correct 50% of deviation per pipe

```
Pipe 1: Correct 50% → Remaining deviation = 50%
Pipe 2: Correct 50% of remaining → Remaining = 25%
Pipe 3: Correct 50% of remaining → Remaining = 12.5%
...continue until near zero
```

### Step 4: Calculate Cylinder Positions
Use forward formula with `pitch_correction` and `yaw_correction`

### Step 5: Verify After Each Pipe
- Measure new pitch/yaw
- Recalculate remaining deviation
- Plan next correction

---

## Example: Correcting to Zero

**Starting**: Pitch = +12.5 mm/m, Yaw = -18.3 mm/m

| Pipe | Current Pitch | Current Yaw | Correction | New Pitch | New Yaw |
|------|---------------|-------------|------------|-----------|---------|
| 1    | +12.50        | -18.30      | -6.25, +9.15 | +6.25    | -9.15   |
| 2    | +6.25         | -9.15       | -3.12, +4.58 | +3.12    | -4.58   |
| 3    | +3.12         | -4.58       | -1.56, +2.29 | +1.56    | -2.29   |
| 4    | +1.56         | -2.29       | -0.78, +1.14 | +0.78    | -1.14   |
| 5    | +0.78         | -1.14       | -0.39, +0.57 | +0.39    | -0.57   |

**Result**: After 5 pipes, deviations reduced to < 1 mm/m

---

## Key Principles

1. ✅ **Progressive Correction**: Don't try to correct everything in one pipe
2. ✅ **Monitor Continuously**: Check after each pipe installation
3. ✅ **Stay Within Limits**: Cylinders must be 0-50mm (or your stroke limit)
4. ✅ **Account for Gradient**: Natural tunnel gradient affects pitch
5. ✅ **Be Patient**: Gradual corrections are more stable than aggressive ones

---

## Correction Per Pipe

```
Correction in mm = steering_rate_mm_per_m × pipe_length_meters

Example:
- Pipe length: 3 meters
- Pitch correction: -10 mm/m
- Actual correction: -10 × 3 = -30 mm over one pipe
```

---

## Common Values

- **Mounting Diameter**: 715 mm → Radius = 0.3575 m
- **Stroke**: 50 mm → Center = 25 mm
- **Pipe Length**: 3000 mm = 3 m
- **Target**: Pitch = 0 mm/m, Yaw = 0 mm/m

---

## Quick Check

**If current pitch = +10 mm/m and you want 0:**
- Required correction = -10 mm/m
- Apply -5 mm/m per pipe (50% strategy)
- After 2 pipes: +2.5 mm/m
- After 3 pipes: +1.25 mm/m
- After 4 pipes: +0.625 mm/m
- After 5 pipes: +0.31 mm/m ✅ Close enough!

---

## Use the Simulator

Run `steering_correction_simulator.py` to see real examples!

