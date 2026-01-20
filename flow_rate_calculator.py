"""
Flow Rate Calculator for Pipe Systems
======================================

Based on Flow Rate Diagram 3.5.3
Calculates flow rate (Q) from pipe diameter (DN) and velocity (V)
or calculates velocity from flow rate and diameter.

Formula: Q = A × V
Where:
  Q = Flow rate (m³/h)
  A = Cross-sectional area (m²)
  V = Velocity (m/s)
"""

import math
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class PipeSpecs:
    """Pipe specifications"""
    nominal_diameter: int  # DN in mm
    inner_diameter_mm: float  # Actual inner diameter in mm
    inner_diameter_m: float  # Actual inner diameter in meters
    
    @property
    def cross_sectional_area_m2(self) -> float:
        """Calculate cross-sectional area in m²"""
        return math.pi * (self.inner_diameter_m / 2) ** 2


# Standard pipe specifications (DN to inner diameter mapping)
# These are approximate - actual values depend on pipe standard (ISO, DIN, etc.)
PIPE_SPECS = {
    50: PipeSpecs(50, 50.0, 0.050),
    80: PipeSpecs(80, 80.0, 0.080),
    100: PipeSpecs(100, 100.0, 0.100),
    125: PipeSpecs(125, 125.0, 0.125),
    150: PipeSpecs(150, 150.0, 0.150),
}


class FlowRateCalculator:
    """
    Flow rate calculator based on diagram 3.5.3
    
    Uses the relationship: Q = A × V
    Where Q is flow rate (m³/h), A is area (m²), V is velocity (m/s)
    """
    
    def __init__(self):
        """Initialize calculator with pipe specifications"""
        self.pipe_specs = PIPE_SPECS
        
        # Data points from diagram at V = 3.5 m/s
        self.diagram_data = {
            50: 24.7,   # m³/h at 3.5 m/s
            80: 63.3,   # m³/h at 3.5 m/s
            100: 98.9,  # m³/h at 3.5 m/s
            125: 154.6, # m³/h at 3.5 m/s
            150: 222.6  # m³/h at 3.5 m/s
        }
    
    def calculate_flow_rate(self, diameter_dn: int, velocity_mps: float) -> float:
        """
        Calculate flow rate from diameter and velocity
        
        Args:
            diameter_dn: Nominal diameter (DN) in mm (50, 80, 100, 125, 150)
            velocity_mps: Velocity in m/s
        
        Returns:
            Flow rate in m³/h
        """
        if diameter_dn not in self.pipe_specs:
            raise ValueError(f"Unsupported diameter DN {diameter_dn}. Supported: {list(self.pipe_specs.keys())}")
        
        pipe = self.pipe_specs[diameter_dn]
        area_m2 = pipe.cross_sectional_area_m2
        
        # Q = A × V
        # Convert from m³/s to m³/h (multiply by 3600)
        flow_rate_m3h = area_m2 * velocity_mps * 3600
        
        return round(flow_rate_m3h, 1)
    
    def calculate_velocity(self, diameter_dn: int, flow_rate_m3h: float) -> float:
        """
        Calculate velocity from diameter and flow rate
        
        Args:
            diameter_dn: Nominal diameter (DN) in mm
            flow_rate_m3h: Flow rate in m³/h
        
        Returns:
            Velocity in m/s
        """
        if diameter_dn not in self.pipe_specs:
            raise ValueError(f"Unsupported diameter DN {diameter_dn}. Supported: {list(self.pipe_specs.keys())}")
        
        pipe = self.pipe_specs[diameter_dn]
        area_m2 = pipe.cross_sectional_area_m2
        
        # V = Q / A
        # Convert from m³/h to m³/s (divide by 3600)
        flow_rate_m3s = flow_rate_m3h / 3600
        velocity_mps = flow_rate_m3s / area_m2
        
        return round(velocity_mps, 2)
    
    def verify_diagram_data(self) -> Dict:
        """
        Verify calculated values match diagram data at V = 3.5 m/s
        
        Returns:
            Comparison of calculated vs diagram values
        """
        results = {}
        test_velocity = 3.5  # m/s
        
        print("="*80)
        print("VERIFYING DIAGRAM DATA (at V = 3.5 m/s)")
        print("="*80)
        print(f"\n{'DN':<8} {'Diagram (m³/h)':<18} {'Calculated (m³/h)':<20} {'Difference':<15} {'Match'}")
        print("-"*80)
        
        for dn in sorted(self.diagram_data.keys()):
            diagram_value = self.diagram_data[dn]
            calculated_value = self.calculate_flow_rate(dn, test_velocity)
            difference = abs(calculated_value - diagram_value)
            match = "✅" if difference < 1.0 else "⚠️"
            
            results[dn] = {
                'diagram': diagram_value,
                'calculated': calculated_value,
                'difference': difference,
                'match': difference < 1.0
            }
            
            print(f"{dn:<8} {diagram_value:<18.1f} {calculated_value:<20.1f} {difference:<15.2f} {match}")
        
        return results
    
    def generate_flow_rate_table(self, velocity_range: Tuple[float, float] = (0, 4.0),
                                 velocity_step: float = 0.5) -> Dict:
        """
        Generate flow rate table for all pipe sizes
        
        Args:
            velocity_range: (min, max) velocity in m/s
            velocity_step: Step size for velocity
        
        Returns:
            Dictionary with flow rates for each DN and velocity
        """
        table = {}
        velocities = []
        v = velocity_range[0]
        while v <= velocity_range[1]:
            velocities.append(round(v, 1))
            v += velocity_step
        
        for dn in sorted(self.pipe_specs.keys()):
            table[dn] = {}
            for vel in velocities:
                table[dn][vel] = self.calculate_flow_rate(dn, vel)
        
        return table
    
    def print_flow_rate_table(self, velocity_range: Tuple[float, float] = (0, 4.0),
                             velocity_step: float = 0.5):
        """Print formatted flow rate table"""
        table = self.generate_flow_rate_table(velocity_range, velocity_step)
        
        print("\n" + "="*80)
        print("FLOW RATE TABLE (Q in m³/h)")
        print("="*80)
        
        # Header
        velocities = sorted(list(table[list(table.keys())[0]].keys()))
        header = f"{'V (m/s)':<12}"
        for dn in sorted(table.keys()):
            header += f"DN {dn:<6}"
        print(header)
        print("-"*80)
        
        # Data rows
        for vel in velocities:
            row = f"{vel:<12.1f}"
            for dn in sorted(table.keys()):
                row += f"{table[dn][vel]:<10.1f}"
            print(row)
    
    def find_diameter_for_flow_rate(self, flow_rate_m3h: float, 
                                   max_velocity_mps: float = 3.5) -> Dict:
        """
        Find appropriate pipe diameter for a given flow rate
        
        Args:
            flow_rate_m3h: Required flow rate in m³/h
            max_velocity_mps: Maximum allowed velocity in m/s (default 3.5)
        
        Returns:
            Dictionary with recommended diameter and calculated velocity
        """
        results = {}
        
        for dn in sorted(self.pipe_specs.keys()):
            velocity = self.calculate_velocity(dn, flow_rate_m3h)
            results[dn] = {
                'velocity_mps': velocity,
                'flow_rate_m3h': flow_rate_m3h,
                'within_limit': velocity <= max_velocity_mps,
                'recommended': velocity <= max_velocity_mps and velocity > 0
            }
        
        # Find best match (smallest diameter that meets velocity requirement)
        recommended = None
        for dn in sorted(results.keys()):
            if results[dn]['within_limit']:
                recommended = dn
                break
        
        return {
            'required_flow_rate': flow_rate_m3h,
            'max_velocity': max_velocity_mps,
            'results_by_diameter': results,
            'recommended_diameter': recommended
        }


def main():
    """Demonstrate flow rate calculator"""
    calc = FlowRateCalculator()
    
    print("="*80)
    print("FLOW RATE CALCULATOR - Based on Diagram 3.5.3")
    print("="*80)
    
    # Verify diagram data
    calc.verify_diagram_data()
    
    # Print flow rate table
    calc.print_flow_rate_table(velocity_range=(0, 4.0), velocity_step=0.5)
    
    # Example calculations
    print("\n" + "="*80)
    print("EXAMPLE CALCULATIONS")
    print("="*80)
    
    print("\n1. Calculate flow rate for DN 100 at 2.5 m/s:")
    q = calc.calculate_flow_rate(100, 2.5)
    print(f"   Flow rate: {q} m³/h")
    
    print("\n2. Calculate velocity for DN 125 with flow rate 100 m³/h:")
    v = calc.calculate_velocity(125, 100)
    print(f"   Velocity: {v} m/s")
    
    print("\n3. Find appropriate diameter for 80 m³/h (max velocity 3.5 m/s):")
    result = calc.find_diameter_for_flow_rate(80, max_velocity_mps=3.5)
    print(f"   Required flow rate: {result['required_flow_rate']} m³/h")
    print(f"   Maximum velocity: {result['max_velocity']} m/s")
    print(f"\n   Results by diameter:")
    for dn in sorted(result['results_by_diameter'].keys()):
        r = result['results_by_diameter'][dn]
        status = "✅ Recommended" if r['recommended'] else ("⚠️ Too fast" if not r['within_limit'] else "OK")
        print(f"     DN {dn}: Velocity = {r['velocity_mps']:.2f} m/s - {status}")
    
    if result['recommended_diameter']:
        print(f"\n   ✅ Recommended: DN {result['recommended_diameter']}")


if __name__ == "__main__":
    main()

