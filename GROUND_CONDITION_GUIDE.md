# Ground Condition Guide for Steering Corrections

## Overview

Ground condition is **CRITICAL** for steering corrections in microtunneling. Different ground types have different maximum steering rates to prevent:
- Jacking pressure increase
- Procedure halting
- Equipment damage

---

## Ground Condition Types

### 1. **Soft Geology**

**Characteristics:**
- Soft clay, loose sand, soft soil
- Low resistance to steering
- Can handle more aggressive corrections

**Steering Limits:**
- **Maximum allowed**: 10 mm/m
- **Recommended maximum**: 8 mm/m
- **Strategy**: Can use more aggressive corrections (50-70% per pipe)

**When to Use:**
- Soft clay deposits
- Loose granular soils
- Low-density ground

---

### 2. **Mixed Geology**

**Characteristics:**
- Combination of soft and hard layers
- Variable resistance
- Moderate steering sensitivity

**Steering Limits:**
- **Maximum allowed**: 4 mm/m
- **Recommended maximum**: 3 mm/m
- **Strategy**: Limit to 2-4 mm/m to avoid jacking pressure increase
- **CRITICAL**: Exceeding limits can cause significant pressure rise

**When to Use:**
- Mixed soil layers
- Alternating soft/hard zones
- Variable ground conditions

**⚠️ WARNING**: 
- Do NOT exceed 4 mm/m total steering rate
- Monitor jacking pressure closely
- Use conservative corrections (30-50% per pipe)

---

### 3. **Rock Geology**

**Characteristics:**
- Hard rock formations
- High resistance to steering
- Very sensitive to corrections

**Steering Limits:**
- **Maximum allowed**: 2 mm/m (ABSOLUTE MAXIMUM)
- **Recommended maximum**: 1.5 mm/m (for safety)
- **Strategy**: Take corrections with utmost care
- **EXTREMELY CRITICAL**: More than 2 mm/m can cause:
  - Immediate jacking pressure increase
  - Procedure halting
  - Equipment damage risk

**When to Use:**
- Rock formations
- Hard consolidated ground
- High-strength geological units

**⚠️ CRITICAL WARNINGS**: 
- **NEVER exceed 2 mm/m** total steering rate
- Use very gradual corrections (20-30% per pipe)
- Monitor jacking pressure continuously
- Be prepared to stop if pressure increases
- Consider splitting corrections over more pipes

---

## How the System Works

### Automatic Limiting

The steering calculator automatically limits corrections based on ground condition:

```python
# Example: Requested correction exceeds rock limit
Requested: Pitch = -8 mm/m, Yaw = +6 mm/m
Total rate: √(8² + 6²) = 10 mm/m

Ground: ROCK (max 2 mm/m)

System automatically scales down:
Scale factor = 2 / 10 = 0.2
Applied: Pitch = -1.6 mm/m, Yaw = +1.2 mm/m
Total rate: 2 mm/m ✅
```

### Warnings Generated

The system generates warnings when:
- Correction exceeds maximum limit (CRITICAL warning)
- Correction exceeds recommended maximum (WARNING)
- Rock ground with rate > 1.5 mm/m (CAUTION)

---

## Correction Strategy by Ground Type

### Soft Ground Strategy

```
Pipe 1: Correct 60% of deviation
Pipe 2: Correct 60% of remaining
Pipe 3: Fine-tune to zero
```

**Example:**
- Initial: Pitch = +15 mm/m
- Pipe 1: Correct -9 mm/m → New: +6 mm/m
- Pipe 2: Correct -3.6 mm/m → New: +2.4 mm/m
- Pipe 3: Correct -2.4 mm/m → New: 0 mm/m ✅

### Mixed Ground Strategy

```
Pipe 1: Correct 40% of deviation (limit to 3 mm/m max)
Pipe 2: Correct 40% of remaining
Pipe 3: Continue gradual correction
Pipe 4: Fine-tune to zero
```

**Example:**
- Initial: Pitch = +12 mm/m
- Pipe 1: Correct -3 mm/m (limited) → New: +9 mm/m
- Pipe 2: Correct -3 mm/m (limited) → New: +6 mm/m
- Pipe 3: Correct -3 mm/m (limited) → New: +3 mm/m
- Pipe 4: Correct -3 mm/m → New: 0 mm/m ✅

### Rock Ground Strategy

```
Pipe 1: Correct 25% of deviation (limit to 1.5 mm/m max)
Pipe 2: Correct 25% of remaining
Pipe 3: Continue very gradual correction
Pipe 4-6: Continue gradual approach
Pipe 7+: Fine-tune to zero
```

**Example:**
- Initial: Pitch = +12 mm/m
- Pipe 1: Correct -1.5 mm/m (limited) → New: +10.5 mm/m
- Pipe 2: Correct -1.5 mm/m (limited) → New: +9 mm/m
- Pipe 3: Correct -1.5 mm/m (limited) → New: +7.5 mm/m
- ... (continues gradually)
- Pipe 8: Correct -1.5 mm/m → New: 0 mm/m ✅

**Note**: Rock corrections take longer but are safer!

---

## Monitoring and Safety

### Key Metrics to Monitor

1. **Jacking Pressure**
   - Monitor continuously during corrections
   - Stop if pressure increases unexpectedly
   - Especially critical in rock ground

2. **Steering Rate**
   - Total rate = √(pitch² + yaw²)
   - Must stay within ground condition limits
   - System will warn if approaching limits

3. **Cylinder Positions**
   - Ensure all cylinders within stroke limits (0-50mm)
   - Check for extreme positions

### Safety Checklist

Before applying corrections:
- [ ] Ground condition identified correctly
- [ ] Maximum steering rate checked
- [ ] Jacking pressure baseline recorded
- [ ] Correction plan reviewed
- [ ] Monitoring plan in place

During corrections:
- [ ] Monitor jacking pressure continuously
- [ ] Check steering rate after each pipe
- [ ] Verify cylinder positions
- [ ] Adjust plan if pressure increases

---

## Examples

### Example 1: Soft Ground

```
Current: Pitch = +10 mm/m, Yaw = -15 mm/m
Ground: SOFT
Max allowed: 10 mm/m

Requested correction: Pitch = -10 mm/m, Yaw = +15 mm/m
Total rate: √(10² + 15²) = 18.03 mm/m

⚠️ Exceeds soft ground limit (10 mm/m)
System scales to: Pitch = -5.55 mm/m, Yaw = +8.32 mm/m
Total rate: 10 mm/m ✅
```

### Example 2: Mixed Ground

```
Current: Pitch = +8 mm/m, Yaw = -6 mm/m
Ground: MIXED
Max allowed: 4 mm/m

Requested correction: Pitch = -8 mm/m, Yaw = +6 mm/m
Total rate: √(8² + 6²) = 10 mm/m

⚠️ Exceeds mixed ground limit (4 mm/m)
System scales to: Pitch = -3.2 mm/m, Yaw = +2.4 mm/m
Total rate: 4 mm/m ✅
```

### Example 3: Rock Ground

```
Current: Pitch = +5 mm/m, Yaw = -3 mm/m
Ground: ROCK
Max allowed: 2 mm/m

Requested correction: Pitch = -5 mm/m, Yaw = +3 mm/m
Total rate: √(5² + 3²) = 5.83 mm/m

⚠️ CRITICAL: Exceeds rock ground limit (2 mm/m)
System scales to: Pitch = -1.72 mm/m, Yaw = +1.03 mm/m
Total rate: 2 mm/m ✅
```

---

## Best Practices

1. **Always specify ground condition** before planning corrections
2. **Start conservative** - you can always correct more later
3. **Monitor jacking pressure** - especially in rock
4. **Use gradual corrections** - better safe than sorry
5. **Adjust strategy** based on actual ground response
6. **Document ground conditions** along the tunnel alignment
7. **Be prepared to stop** if pressure increases unexpectedly

---

## Summary Table

| Ground Type | Max Rate | Recommended | Strategy | Risk Level |
|-------------|----------|-------------|----------|------------|
| **Soft** | 10 mm/m | 8 mm/m | Aggressive (50-70%) | Low |
| **Mixed** | 4 mm/m | 3 mm/m | Moderate (30-50%) | Medium |
| **Rock** | 2 mm/m | 1.5 mm/m | Very gradual (20-30%) | **High** |

---

## Remember

> **"In rock ground, more than 2 mm/m steering can halt the jacking procedure. Take corrections with utmost care!"**

Always prioritize safety over speed. Gradual corrections are better than aggressive ones that cause problems.

