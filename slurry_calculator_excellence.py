"""
Slurry Calculation Calculator
=============================

Based on Excellence Pump Industry Co., Ltd. Slurry Pumping Manual

Implements all formulas for calculating slurry properties:
- S: SG of solids
- Sw: SG of liquid
- Sm: SG of mixture
- Cw: Concentration by weight (%)
- Cv: Concentration by volume (%)

When any 3 variables are known, the other 2 can be calculated.
"""

import math
from typing import Optional, Dict, Tuple
from dataclasses import dataclass


@dataclass
class SlurryProperties:
    """Container for slurry properties"""
    S: Optional[float] = None   # SG of solids
    Sw: Optional[float] = None  # SG of liquid (water = 1.0)
    Sm: Optional[float] = None  # SG of mixture
    Cw: Optional[float] = None  # Concentration by weight (0-1, not %)
    Cv: Optional[float] = None  # Concentration by volume (0-1, not %)
    
    def count_known(self) -> int:
        """Count how many properties are known"""
        return sum([
            self.S is not None,
            self.Sw is not None,
            self.Sm is not None,
            self.Cw is not None,
            self.Cv is not None
        ])
    
    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert to dictionary"""
        return {
            'S': self.S,
            'Sw': self.Sw,
            'Sm': self.Sm,
            'Cw': self.Cw,
            'Cv': self.Cv
        }


class SlurryCalculator:
    """
    Calculator for slurry properties based on Excellence Pump formulas.
    
    All formulas from the Excellence Pump Industry Co., Ltd. manual.
    """
    
    def __init__(self):
        """Initialize calculator"""
        pass
    
    def calculate_sw(self, S: float, Sm: float, Cw: float, Cv: float) -> float:
        """
        Calculate Sw (SG of liquid) from other variables.
        
        Uses three alternative formulas and returns average for accuracy.
        
        Args:
            S: SG of solids
            Sm: SG of mixture
            Cw: Concentration by weight (0-1)
            Cv: Concentration by volume (0-1)
        
        Returns:
            Sw: SG of liquid
        """
        # Formula 1: Sw = S(Sm·Cw - Sm) / (Sm·Cw - S)
        try:
            sw1 = S * (Sm * Cw - Sm) / (Sm * Cw - S)
        except (ZeroDivisionError, ValueError):
            sw1 = None
        
        # Formula 2: Sw = (S·Cv - Sm) / (Cv - 1)
        try:
            sw2 = (S * Cv - Sm) / (Cv - 1)
        except (ZeroDivisionError, ValueError):
            sw2 = None
        
        # Formula 3: Sw = S[Cv(Cw - 1)] / [Cw(Cv - 1)]
        try:
            sw3 = S * (Cv * (Cw - 1)) / (Cw * (Cv - 1))
        except (ZeroDivisionError, ValueError):
            sw3 = None
        
        # Return average of valid formulas
        values = [v for v in [sw1, sw2, sw3] if v is not None and v > 0]
        if not values:
            raise ValueError("Cannot calculate Sw with given parameters")
        return sum(values) / len(values)
    
    def calculate_s(self, Sw: float, Sm: float, Cw: float, Cv: float) -> float:
        """
        Calculate S (SG of solids) from other variables.
        
        Args:
            Sw: SG of liquid
            Sm: SG of mixture
            Cw: Concentration by weight (0-1)
            Cv: Concentration by volume (0-1)
        
        Returns:
            S: SG of solids
        """
        # Formula 1: S = Sw·Cw(Cv - 1) / [Cv(Cw - 1)]
        try:
            s1 = Sw * Cw * (Cv - 1) / (Cv * (Cw - 1))
        except (ZeroDivisionError, ValueError):
            s1 = None
        
        # Formula 2: S = Sw + (Sm - Sw) / Cv
        try:
            s2 = Sw + (Sm - Sw) / Cv
        except (ZeroDivisionError, ValueError):
            s2 = None
        
        # Formula 3: S = Sw·Cw / (Cw - 1 + Sw/Sm)
        try:
            s3 = Sw * Cw / (Cw - 1 + Sw / Sm)
        except (ZeroDivisionError, ValueError):
            s3 = None
        
        values = [v for v in [s1, s2, s3] if v is not None and v > 0]
        if not values:
            raise ValueError("Cannot calculate S with given parameters")
        return sum(values) / len(values)
    
    def calculate_sm(self, S: float, Sw: float, Cw: float, Cv: float) -> float:
        """
        Calculate Sm (SG of mixture) from other variables.
        
        Args:
            S: SG of solids
            Sw: SG of liquid
            Cw: Concentration by weight (0-1)
            Cv: Concentration by volume (0-1)
        
        Returns:
            Sm: SG of mixture
        """
        # Formula 1: Sm = Sw / [1 - Cw(1 - Sw/S)]
        try:
            sm1 = Sw / (1 - Cw * (1 - Sw / S))
        except (ZeroDivisionError, ValueError):
            sm1 = None
        
        # Formula 2: Sm = Sw + Cv(S - Sw)
        try:
            sm2 = Sw + Cv * (S - Sw)
        except (ZeroDivisionError, ValueError):
            sm2 = None
        
        # Formula 3: Sm = Sw(Cv - 1) / (Cw - 1)
        try:
            sm3 = Sw * (Cv - 1) / (Cw - 1)
        except (ZeroDivisionError, ValueError):
            sm3 = None
        
        values = [v for v in [sm1, sm2, sm3] if v is not None and v > 0]
        if not values:
            raise ValueError("Cannot calculate Sm with given parameters")
        return sum(values) / len(values)
    
    def calculate_cw(self, S: float, Sw: float, Sm: float, Cv: float) -> float:
        """
        Calculate Cw (Concentration by weight) from other variables.
        
        Args:
            S: SG of solids
            Sw: SG of liquid
            Sm: SG of mixture
            Cv: Concentration by volume (0-1)
        
        Returns:
            Cw: Concentration by weight (0-1)
        """
        # Formula 1: Cw = S(Sm - Sw) / [Sm(S - Sw)]
        try:
            cw1 = S * (Sm - Sw) / (Sm * (S - Sw))
        except (ZeroDivisionError, ValueError):
            cw1 = None
        
        # Formula 2: Cw = S·Cv / [Sw + Cv(S - Sw)]
        try:
            cw2 = S * Cv / (Sw + Cv * (S - Sw))
        except (ZeroDivisionError, ValueError):
            cw2 = None
        
        # Formula 3: Cw = 1 + Sw(Cv - 1) / Sm
        try:
            cw3 = 1 + Sw * (Cv - 1) / Sm
        except (ZeroDivisionError, ValueError):
            cw3 = None
        
        values = [v for v in [cw1, cw2, cw3] if v is not None and 0 <= v <= 1]
        if not values:
            raise ValueError("Cannot calculate Cw with given parameters")
        return sum(values) / len(values)
    
    def calculate_cv(self, S: float, Sw: float, Sm: float, Cw: float) -> float:
        """
        Calculate Cv (Concentration by volume) from other variables.
        
        Args:
            S: SG of solids
            Sw: SG of liquid
            Sm: SG of mixture
            Cw: Concentration by weight (0-1)
        
        Returns:
            Cv: Concentration by volume (0-1)
        """
        # Formula 1: Cv = (Sm - Sw) / (S - Sw)
        try:
            cv1 = (Sm - Sw) / (S - Sw)
        except (ZeroDivisionError, ValueError):
            cv1 = None
        
        # Formula 2: Cv = Sw / (Sw - S + S/Cw)
        try:
            cv2 = Sw / (Sw - S + S / Cw)
        except (ZeroDivisionError, ValueError):
            cv2 = None
        
        # Formula 3: Cv = 1 + Sm(Cw - 1) / Sw
        try:
            cv3 = 1 + Sm * (Cw - 1) / Sw
        except (ZeroDivisionError, ValueError):
            cv3 = None
        
        values = [v for v in [cv1, cv2, cv3] if v is not None and 0 <= v <= 1]
        if not values:
            raise ValueError("Cannot calculate Cv with given parameters")
        return sum(values) / len(values)
    
    def verify_special_relationship(self, S: float, Sm: float, Cw: float, Cv: float) -> Tuple[float, float, bool]:
        """
        Verify the special relationship: Cw/Cv = S/Sm
        
        Args:
            S: SG of solids
            Sm: SG of mixture
            Cw: Concentration by weight (0-1)
            Cv: Concentration by volume (0-1)
        
        Returns:
            Tuple of (Cw/Cv, S/Sm, match_status)
        """
        ratio1 = Cw / Cv
        ratio2 = S / Sm
        match = abs(ratio1 - ratio2) < 0.001  # Allow small floating point error
        return ratio1, ratio2, match
    
    def solve(self, props: SlurryProperties) -> SlurryProperties:
        """
        Solve for unknown properties when at least 3 are known.
        
        Args:
            props: SlurryProperties with at least 3 known values
        
        Returns:
            SlurryProperties with all values calculated
        """
        known_count = props.count_known()
        if known_count < 3:
            raise ValueError(f"Need at least 3 known properties, got {known_count}")
        
        result = SlurryProperties(
            S=props.S,
            Sw=props.Sw,
            Sm=props.Sm,
            Cw=props.Cw,
            Cv=props.Cv
        )
        
        # Iterate until all are calculated
        max_iterations = 10
        for iteration in range(max_iterations):
            if result.count_known() == 5:
                break
            
            previous_count = result.count_known()
            
            # Calculate Sm if missing (can use S, Sw, Cv - simplest formula)
            if result.Sm is None:
                try:
                    if all([result.S, result.Sw, result.Cv]):
                        # Formula 2: Sm = Sw + Cv(S - Sw) - only needs 3 variables!
                        result.Sm = result.Sw + result.Cv * (result.S - result.Sw)
                    elif all([result.S, result.Sw, result.Cw]):
                        # Formula 1: Sm = Sw / [1 - Cw(1 - Sw/S)]
                        result.Sm = result.Sw / (1 - result.Cw * (1 - result.Sw / result.S))
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Calculate Cv if missing (can use S, Sw, Sm - simplest formula)
            if result.Cv is None:
                try:
                    if all([result.S, result.Sw, result.Sm]):
                        # Formula 1: Cv = (Sm - Sw) / (S - Sw) - only needs 3 variables!
                        result.Cv = (result.Sm - result.Sw) / (result.S - result.Sw)
                    elif all([result.S, result.Sw, result.Cw]):
                        # Formula 2: Cv = Sw / (Sw - S + S/Cw)
                        result.Cv = result.Sw / (result.Sw - result.S + result.S / result.Cw)
                    elif all([result.Sw, result.Sm, result.Cw]):
                        # Formula 3: Cv = 1 + Sm(Cw - 1) / Sw
                        result.Cv = 1 + result.Sm * (result.Cw - 1) / result.Sw
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Calculate Cw if missing
            if result.Cw is None:
                try:
                    if all([result.S, result.Sw, result.Sm, result.Cv]):
                        result.Cw = self.calculate_cw(result.S, result.Sw, result.Sm, result.Cv)
                    elif all([result.S, result.Sm, result.Cv]):
                        # Use special relationship: Cw/Cv = S/Sm, so Cw = (S/Sm) * Cv
                        result.Cw = (result.S / result.Sm) * result.Cv
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Calculate S if missing
            if result.S is None:
                try:
                    if all([result.Sw, result.Sm, result.Cw, result.Cv]):
                        result.S = self.calculate_s(result.Sw, result.Sm, result.Cw, result.Cv)
                except (ValueError, ZeroDivisionError):
                    pass
            
            # Calculate Sw if missing
            if result.Sw is None:
                try:
                    if all([result.S, result.Sm, result.Cw, result.Cv]):
                        result.Sw = self.calculate_sw(result.S, result.Sm, result.Cw, result.Cv)
                except (ValueError, ZeroDivisionError):
                    pass
            
            # If no progress made, break
            if result.count_known() == previous_count:
                break
        
        if result.count_known() < 5:
            raise ValueError(f"Could not solve for all properties. Only {result.count_known()}/5 calculated.")
        
        return result


def main():
    """Demonstrate slurry calculator"""
    calc = SlurryCalculator()
    
    print("="*80)
    print("SLURRY CALCULATOR - Excellence Pump Formulas")
    print("="*80)
    
    # Example 1: Calculate mixture properties
    print("\nExample 1: Calculate Mixture SG from Solids and Concentration")
    print("-"*80)
    props1 = SlurryProperties(
        S=2.65,      # Silica
        Sw=1.0,      # Water
        Cv=0.30      # 30% by volume
    )
    result1 = calc.solve(props1)
    print(f"Input:")
    print(f"  S (SG solids): {props1.S}")
    print(f"  Sw (SG liquid): {props1.Sw}")
    print(f"  Cv (vol %): {props1.Cv*100:.1f}%")
    print(f"\nCalculated:")
    print(f"  Sm (SG mixture): {result1.Sm:.3f}")
    print(f"  Cw (weight %): {result1.Cw*100:.1f}%")
    
    # Verify special relationship
    ratio1, ratio2, match = calc.verify_special_relationship(
        result1.S, result1.Sm, result1.Cw, result1.Cv
    )
    print(f"\nSpecial Relationship Check:")
    print(f"  Cw/Cv = {ratio1:.4f}")
    print(f"  S/Sm = {ratio2:.4f}")
    print(f"  Match: {'YES' if match else 'NO'}")
    
    # Example 2: Calculate concentration from mixture SG
    print("\n" + "="*80)
    print("Example 2: Calculate Concentration from Mixture SG")
    print("-"*80)
    props2 = SlurryProperties(
        S=2.65,      # Silica
        Sw=1.0,      # Water
        Sm=1.495     # Known mixture SG
    )
    result2 = calc.solve(props2)
    print(f"Input:")
    print(f"  S (SG solids): {props2.S}")
    print(f"  Sw (SG liquid): {props2.Sw}")
    print(f"  Sm (SG mixture): {props2.Sm}")
    print(f"\nCalculated:")
    print(f"  Cw (weight %): {result2.Cw*100:.1f}%")
    print(f"  Cv (vol %): {result2.Cv*100:.1f}%")
    
    # Example 3: Calculate from weight concentration
    print("\n" + "="*80)
    print("Example 3: Calculate from Weight Concentration")
    print("-"*80)
    props3 = SlurryProperties(
        S=2.65,      # Silica
        Sw=1.0,      # Water
        Cw=0.50      # 50% by weight
    )
    result3 = calc.solve(props3)
    print(f"Input:")
    print(f"  S (SG solids): {props3.S}")
    print(f"  Sw (SG liquid): {props3.Sw}")
    print(f"  Cw (weight %): {props3.Cw*100:.1f}%")
    print(f"\nCalculated:")
    print(f"  Sm (SG mixture): {result3.Sm:.3f}")
    print(f"  Cv (vol %): {result3.Cv*100:.1f}%")
    
    print("\n" + "="*80)
    print("Calculator Ready for Use!")
    print("="*80)


if __name__ == "__main__":
    main()

