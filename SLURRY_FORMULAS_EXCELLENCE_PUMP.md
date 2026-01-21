# Slurry Calculation Formulas - Excellence Pump Manual

## Source
Excellence Pump Industry Co., Ltd. - Slurry Pumping Manual

---

## Basic Concepts

### States of Matter
- **Three states:** Gaseous, Liquid, Solid
- **Fluids:** Liquids and gases (can flow and take shape of container)
- **Liquids:** Virtually incompressible, can have separating surfaces
- **Gases:** Compressible, mix with each other

### Density
- **Symbol:** ρ [kg/m³]
- **Alternative units:** [t/m³] or [kg/L] (numerically equal)
- **Definition:** Mass per unit volume

### Specific Gravity (SG)
- **Definition:** Dimensionless number = Density of material / Density of water
- **Reference:** Water density = 1000 kg/m³
- **Formula:** `SG = ρ / 1000`
- **Examples:**
  - Water: SG = 1.0
  - Silica (sand/rock): SG = 2.65

---

## Slurry Variables

A **slurry** is a mixture of liquid and solid particles.

### Five Inter-Related Variables

| Symbol | Name | Description | Units |
|--------|------|-------------|-------|
| **S** | SG of solids | Specific gravity of solid particles | Dimensionless |
| **Sw** | SG of liquid | Specific gravity of liquid carrier (water = 1) | Dimensionless |
| **Sm** | SG of mixture | Specific gravity of slurry mixture | Dimensionless |
| **Cw** | Concentration by weight | Percentage of solids by weight | [%] |
| **Cv** | Concentration by volume | Percentage of solids by true volume | [%] |

### Important Notes
- **Cw and Cv:** Divide by 100 if expressed as percentages before using in equations
- **True volume:** Accounts for packing efficiency (typically 50-80% for different particle shapes/sizes)
- **Loose sand:** Cv ≈ 73% (packing efficiency)

---

## Slurry Formula Equations

### Rule
**When any three of the five variables are known, the other two can be calculated.**

---

### 1. Calculate Sw (SG of Liquid)

Given: S, Sm, Cw, Cv

**Formula 1:**
```
Sw = S(Sm·Cw - Sm) / (Sm·Cw - S)
```

**Formula 2:**
```
Sw = (S·Cv - Sm) / (Cv - 1)
```

**Formula 3:**
```
Sw = S[Cv(Cw - 1)] / [Cw(Cv - 1)]
```

**For water:** Sw = 1.0

---

### 2. Calculate S (SG of Solids)

Given: Sw, Sm, Cw, Cv

**Formula 1:**
```
S = Sw·Cw(Cv - 1) / [Cv(Cw - 1)]
```

**Formula 2:**
```
S = Sw + (Sm - Sw) / Cv
```

**Formula 3:**
```
S = Sw·Cw / (Cw - 1 + Sw/Sm)
```

---

### 3. Calculate Sm (SG of Mixture)

Given: Sw, S, Cw, Cv

**Formula 1:**
```
Sm = Sw / [1 - Cw(1 - Sw/S)]
```

**Formula 2:**
```
Sm = Sw + Cv(S - Sw)
```

**Formula 3:**
```
Sm = Sw(Cv - 1) / (Cw - 1)
```

---

### 4. Calculate Cw (Concentration by Weight)

Given: S, Sw, Sm, Cv

**Formula 1:**
```
Cw = S(Sm - Sw) / [Sm(S - Sw)]
```

**Formula 2:**
```
Cw = S·Cv / [Sw + Cv(S - Sw)]
```

**Formula 3:**
```
Cw = 1 + Sw(Cv - 1) / Sm
```

**Note:** Result is a decimal (0-1). Multiply by 100 for percentage.

---

### 5. Calculate Cv (Concentration by Volume)

Given: S, Sw, Sm, Cw

**Formula 1:**
```
Cv = (Sm - Sw) / (S - Sw)
```

**Formula 2:**
```
Cv = Sw / (Sw - S + S/Cw)
```

**Formula 3:**
```
Cv = 1 + Sm(Cw - 1) / Sw
```

**Note:** Result is a decimal (0-1). Multiply by 100 for percentage.

---

## Special Relationship Formula

### Universal Ratio Equation

**Formula:**
```
Cw / Cv = S / Sm
```

**Key Properties:**
- **Independent of liquid density (Sw)**
- Holds true for solids mixed with **any liquid**
- If Cw/Cv ratio is fixed, S/Sm must have the same value
- Sw can be changed without affecting this ratio

**Use:** Calculate any one of the four variables (Cw, Cv, S, Sm) when the other three are known.

---

## Example Calculations

### Example 1: Calculate Mixture SG
**Given:**
- S = 2.65 (silica)
- Sw = 1.0 (water)
- Cv = 0.30 (30% by volume)

**Calculate Sm:**
```
Sm = Sw + Cv(S - Sw)
Sm = 1.0 + 0.30(2.65 - 1.0)
Sm = 1.0 + 0.30(1.65)
Sm = 1.0 + 0.495
Sm = 1.495
```

### Example 2: Calculate Concentration by Weight
**Given:**
- S = 2.65
- Sw = 1.0
- Sm = 1.495

**Calculate Cw:**
```
Cw = S(Sm - Sw) / [Sm(S - Sw)]
Cw = 2.65(1.495 - 1.0) / [1.495(2.65 - 1.0)]
Cw = 2.65(0.495) / [1.495(1.65)]
Cw = 1.312 / 2.467
Cw = 0.532 (or 53.2%)
```

### Example 3: Verify Special Relationship
**Given:**
- Cw = 0.532
- Cv = 0.30
- S = 2.65
- Sm = 1.495

**Verify:**
```
Cw/Cv = 0.532/0.30 = 1.773
S/Sm = 2.65/1.495 = 1.773
✅ Match!
```

---

## Reference Density Values

| Material | Density [kg/m³] | SG |
|----------|----------------|-----|
| Water (0°C) | 999 | 0.999 |
| Water (100°C) | 957 | 0.957 |
| Ice (0°C) | 895 | 0.895 |
| Steam (100°C) | 0.590 | 0.00059 |
| Silica (sand/rock) | 2650 | 2.65 |

---

## Important Notes

1. **Unit Conversion:** Always divide Cw and Cv by 100 if given as percentages before using in equations
2. **Packing Efficiency:** True volume (Cv) accounts for particle packing (typically 50-80%)
3. **Water Reference:** For water, use Sw = 1.0
4. **Mixture as Fluid:** Slurries can often be treated as equivalent fluids with SG = Sm
5. **Three Variables Rule:** Need at least 3 of 5 variables to calculate the others

---

*Source: Excellence Pump Industry Co., Ltd. Slurry Pumping Manual*

