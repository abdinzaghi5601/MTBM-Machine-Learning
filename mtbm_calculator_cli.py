#!/usr/bin/env python3
"""
MTBM Unified Calculator CLI
============================
Unified command-line interface for all microtunneling calculations:
- Slurry Properties (Excellence Pump formulas)
- Slurry Flow Requirements
- Critical/Settling Velocity
- Excavation Mass Balance
- Bentonite Injection
- Pipejacking Force & Speed
- Power & Cable Sizing

Usage:
    python mtbm_calculator_cli.py              # Interactive mode
    python mtbm_calculator_cli.py --quick      # Quick calculation mode
    python mtbm_calculator_cli.py --help       # Show help
"""

import sys
import math
from typing import Optional, Dict, Any
from dataclasses import dataclass

# Import our calculator modules
from slurry_calculator_excellence import SlurryCalculator, SlurryProperties
from slurry_flow_complete import (
    SlurryFlowCalculator, MachineParams, SlurryPipeParams, SoilParams,
    ParticleParams, SlurryCircuitParams, SOIL_PRESETS, BENTONITE_INJECTION
)
from pipejacking_calculator import (
    PipejackingCalculator, JackingStationParams, SlurryParams, PowerParams,
    PipeParams, TunnelParams, ExpanderConfig
)


# =============================================================================
# DISPLAY HELPERS
# =============================================================================

def clear_screen():
    """Clear terminal screen"""
    print("\033[2J\033[H", end="")


def print_header(title: str, width: int = 70):
    """Print formatted header"""
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str, width: int = 70):
    """Print formatted subheader"""
    print("\n" + "-" * width)
    print(f"  {title}")
    print("-" * width)


def print_menu(title: str, options: list, width: int = 70):
    """Print a menu with numbered options"""
    print_header(title, width)
    for i, option in enumerate(options, 1):
        print(f"  [{i}] {option}")
    print(f"  [0] Back / Exit")
    print("-" * width)


def get_float(prompt: str, default: Optional[float] = None,
              min_val: Optional[float] = None, max_val: Optional[float] = None) -> float:
    """Get float input with validation"""
    while True:
        try:
            default_str = f" [{default}]" if default is not None else ""
            user_input = input(f"  {prompt}{default_str}: ").strip()

            if user_input == "" and default is not None:
                return default

            value = float(user_input)

            if min_val is not None and value < min_val:
                print(f"    Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"    Value must be <= {max_val}")
                continue

            return value
        except ValueError:
            print("    Please enter a valid number")


def get_int(prompt: str, default: Optional[int] = None,
            min_val: Optional[int] = None, max_val: Optional[int] = None) -> int:
    """Get integer input with validation"""
    while True:
        try:
            default_str = f" [{default}]" if default is not None else ""
            user_input = input(f"  {prompt}{default_str}: ").strip()

            if user_input == "" and default is not None:
                return default

            value = int(user_input)

            if min_val is not None and value < min_val:
                print(f"    Value must be >= {min_val}")
                continue
            if max_val is not None and value > max_val:
                print(f"    Value must be <= {max_val}")
                continue

            return value
        except ValueError:
            print("    Please enter a valid integer")


def get_choice(prompt: str, options: list) -> int:
    """Get menu choice"""
    while True:
        try:
            choice = int(input(f"  {prompt}: "))
            if 0 <= choice <= len(options):
                return choice
            print(f"    Please enter 0-{len(options)}")
        except ValueError:
            print("    Please enter a number")


def get_string(prompt: str, default: Optional[str] = None,
               options: Optional[list] = None) -> str:
    """Get string input with optional validation"""
    while True:
        default_str = f" [{default}]" if default is not None else ""
        user_input = input(f"  {prompt}{default_str}: ").strip()

        if user_input == "" and default is not None:
            return default

        if options is not None:
            if user_input.lower() in [o.lower() for o in options]:
                return user_input.lower()
            print(f"    Valid options: {', '.join(options)}")
            continue

        return user_input


def pause():
    """Pause for user to read output"""
    input("\n  Press Enter to continue...")


# =============================================================================
# SLURRY PROPERTIES CALCULATOR
# =============================================================================

def slurry_properties_menu():
    """Slurry properties calculator (Excellence Pump formulas)"""
    calc = SlurryCalculator()

    while True:
        print_menu("SLURRY PROPERTIES CALCULATOR", [
            "Calculate from S, Sw, Cv (volume concentration)",
            "Calculate from S, Sw, Cw (weight concentration)",
            "Calculate from S, Sw, Sm (mixture SG)",
            "Enter any 3 known values",
            "Show formulas reference"
        ])

        choice = get_choice("Select option", range(6))

        if choice == 0:
            return

        elif choice == 1:
            # Calculate from S, Sw, Cv
            print_subheader("Calculate from S, Sw, Cv")
            S = get_float("S (SG of solids)", default=2.65, min_val=1.0, max_val=10.0)
            Sw = get_float("Sw (SG of liquid)", default=1.0, min_val=0.5, max_val=2.0)
            Cv = get_float("Cv (volume concentration, 0-1 or %)", default=0.25, min_val=0, max_val=100)

            # Handle percentage input
            if Cv > 1:
                Cv = Cv / 100

            props = SlurryProperties(S=S, Sw=Sw, Cv=Cv)
            result = calc.solve(props)

            print_subheader("RESULTS")
            print(f"""
  Input:
    S (SG solids):        {S:.3f}
    Sw (SG liquid):       {Sw:.3f}
    Cv (vol conc):        {Cv*100:.1f}%

  Calculated:
    Sm (SG mixture):      {result.Sm:.3f}
    Cw (weight conc):     {result.Cw*100:.1f}%

  Verification (Cw/Cv = S/Sm):
    Cw/Cv = {result.Cw/result.Cv:.4f}
    S/Sm  = {S/result.Sm:.4f}
""")
            pause()

        elif choice == 2:
            # Calculate from S, Sw, Cw
            print_subheader("Calculate from S, Sw, Cw")
            S = get_float("S (SG of solids)", default=2.65, min_val=1.0, max_val=10.0)
            Sw = get_float("Sw (SG of liquid)", default=1.0, min_val=0.5, max_val=2.0)
            Cw = get_float("Cw (weight concentration, 0-1 or %)", default=0.50, min_val=0, max_val=100)

            if Cw > 1:
                Cw = Cw / 100

            props = SlurryProperties(S=S, Sw=Sw, Cw=Cw)
            result = calc.solve(props)

            print_subheader("RESULTS")
            print(f"""
  Input:
    S (SG solids):        {S:.3f}
    Sw (SG liquid):       {Sw:.3f}
    Cw (weight conc):     {Cw*100:.1f}%

  Calculated:
    Sm (SG mixture):      {result.Sm:.3f}
    Cv (vol conc):        {result.Cv*100:.1f}%
""")
            pause()

        elif choice == 3:
            # Calculate from S, Sw, Sm
            print_subheader("Calculate from S, Sw, Sm")
            S = get_float("S (SG of solids)", default=2.65, min_val=1.0, max_val=10.0)
            Sw = get_float("Sw (SG of liquid)", default=1.0, min_val=0.5, max_val=2.0)
            Sm = get_float("Sm (SG of mixture)", default=1.5, min_val=1.0, max_val=5.0)

            props = SlurryProperties(S=S, Sw=Sw, Sm=Sm)
            result = calc.solve(props)

            print_subheader("RESULTS")
            print(f"""
  Input:
    S (SG solids):        {S:.3f}
    Sw (SG liquid):       {Sw:.3f}
    Sm (SG mixture):      {Sm:.3f}

  Calculated:
    Cv (vol conc):        {result.Cv*100:.1f}%
    Cw (weight conc):     {result.Cw*100:.1f}%
""")
            pause()

        elif choice == 4:
            # Enter any 3 known values
            print_subheader("Enter Any 3 Known Values")
            print("  Leave blank if unknown, enter at least 3 values")
            print()

            S = input("  S (SG of solids): ").strip()
            Sw = input("  Sw (SG of liquid): ").strip()
            Sm = input("  Sm (SG of mixture): ").strip()
            Cw = input("  Cw (weight concentration, 0-1 or %): ").strip()
            Cv = input("  Cv (volume concentration, 0-1 or %): ").strip()

            props = SlurryProperties(
                S=float(S) if S else None,
                Sw=float(Sw) if Sw else None,
                Sm=float(Sm) if Sm else None,
                Cw=float(Cw)/100 if Cw and float(Cw) > 1 else (float(Cw) if Cw else None),
                Cv=float(Cv)/100 if Cv and float(Cv) > 1 else (float(Cv) if Cv else None)
            )

            if props.count_known() < 3:
                print("\n  ERROR: Need at least 3 known values")
                pause()
                continue

            try:
                result = calc.solve(props)
                print_subheader("RESULTS")
                print(f"""
  S (SG solids):        {result.S:.3f}
  Sw (SG liquid):       {result.Sw:.3f}
  Sm (SG mixture):      {result.Sm:.3f}
  Cv (vol conc):        {result.Cv*100:.1f}%
  Cw (weight conc):     {result.Cw*100:.1f}%
""")
            except Exception as e:
                print(f"\n  ERROR: {e}")
            pause()

        elif choice == 5:
            # Show formulas
            print_header("SLURRY FORMULAS REFERENCE")
            print("""
  VARIABLES:
  ─────────────────────────────────────────────────────────────
  S   = Specific gravity of solids (e.g., silica = 2.65)
  Sw  = Specific gravity of liquid (water = 1.0)
  Sm  = Specific gravity of mixture
  Cw  = Concentration by weight (decimal 0-1)
  Cv  = Concentration by volume (decimal 0-1)

  KEY FORMULAS:
  ─────────────────────────────────────────────────────────────
  Sm = Sw + Cv(S - Sw)           [Mixture SG from vol conc]

  Cv = (Sm - Sw) / (S - Sw)      [Vol conc from mixture SG]

  Cw = S × Cv / Sm               [Weight conc from vol conc]

  Sm = Sw / [1 - Cw(1 - Sw/S)]   [Mixture SG from weight conc]

  SPECIAL RELATIONSHIP:
  ─────────────────────────────────────────────────────────────
  Cw / Cv = S / Sm               [Always true, independent of Sw]

  When any 3 of 5 variables are known, the other 2 can be calculated.
""")
            pause()


# =============================================================================
# SLURRY FLOW CALCULATOR
# =============================================================================

def slurry_flow_menu():
    """Slurry flow calculator menu"""
    calc = SlurryFlowCalculator()

    while True:
        print_menu("SLURRY FLOW CALCULATOR", [
            "Full excavation & flow calculation",
            "Settling velocity (particle)",
            "Critical deposition velocity",
            "Excavation mass balance",
            "Over-excavation check",
            "Bentonite injection requirements",
            "Slurry pipeline head loss"
        ])

        choice = get_choice("Select option", range(8))

        if choice == 0:
            return

        elif choice == 1:
            # Full calculation
            print_subheader("MACHINE PARAMETERS")
            cutter_dia = get_float("Cutter head diameter (mm)", default=1500, min_val=100)
            pipe_od = get_float("Pipe outer diameter (mm)", default=1400, min_val=100)
            advance_rate = get_float("Advance rate (mm/min)", default=30, min_val=1)

            print_subheader("SLURRY PIPE PARAMETERS")
            feed_dia = get_float("Feed pipe diameter (mm)", default=150, min_val=50)
            return_dia = get_float("Return pipe diameter (mm)", default=150, min_val=50)

            print_subheader("SOIL TYPE")
            print("  Options: sand, clay, gravel, mixed, rock")
            soil_type = get_string("Soil type", default="sand",
                                   options=["sand", "clay", "gravel", "mixed", "rock"])

            pipe_length = get_float("Total pipe length (m)", default=200, min_val=10)

            machine = MachineParams(
                cutter_diameter_mm=cutter_dia,
                pipe_od_mm=pipe_od,
                advance_rate_mm_min=advance_rate
            )
            slurry_pipe = SlurryPipeParams(
                feed_pipe_diameter_mm=feed_dia,
                return_pipe_diameter_mm=return_dia
            )

            results = calc.full_calculation(machine, slurry_pipe, soil_type, pipe_length)

            exc = results['excavation']
            slurry = results['slurry_requirements']
            pipes = results['pipe_verification']
            pump = results['pump_requirements']

            print_header("CALCULATION RESULTS")
            print(f"""
  EXCAVATION:
  ─────────────────────────────────────────────────────────────
  Cutter area:            {exc['cutter_area_m2']:.4f} m²
  Excavation rate:        {exc['excavation_m3_hr']:.3f} m³/hr
  Overcut:                {exc['overcut_mm']:.1f} mm per side

  SLURRY REQUIREMENTS ({slurry['soil_type']}):
  ─────────────────────────────────────────────────────────────
  Bulked volume:          {slurry['bulked_volume_m3_hr']:.3f} m³/hr
  Required flow:          {slurry['required_flow_m3_hr']:.2f} m³/hr
  Recommended flow:       {slurry['recommended_flow_m3_hr']:.2f} m³/hr ({slurry['recommended_flow_L_min']:.0f} L/min)

  PIPE VERIFICATION:
  ─────────────────────────────────────────────────────────────
  Feed velocity:          {pipes['feed']['velocity_mps']:.2f} m/s - {pipes['feed']['velocity_status']}
  Return velocity:        {pipes['return']['velocity_mps']:.2f} m/s - {pipes['return']['velocity_status']}
  Required velocity:      {slurry['min_velocity_mps']:.1f} - 5.0 m/s

  PUMP REQUIREMENTS:
  ─────────────────────────────────────────────────────────────
  Total head:             {pump['total_head_m']:.2f} m
  Motor power:            {pump['motor_power_kW']:.2f} kW
""")
            pause()

        elif choice == 2:
            # Settling velocity
            print_subheader("PARTICLE SETTLING VELOCITY")
            d50 = get_float("Particle d50 (mm)", default=0.5, min_val=0.01, max_val=100)
            sg = get_float("Particle SG", default=2.65, min_val=1.1, max_val=10)
            shape = get_float("Shape factor (0.5-1.0)", default=0.7, min_val=0.3, max_val=1.0)

            particle = ParticleParams(d50_mm=d50, particle_sg=sg, shape_factor=shape)
            result = calc.calculate_settling_velocity(particle)

            print_subheader("RESULTS")
            print(f"""
  Particle diameter:      {result['particle_diameter_mm']:.2f} mm
  Particle SG:            {result['particle_sg']:.2f}
  Shape factor:           {result['shape_factor_applied']:.2f}

  Settling velocity:      {result['settling_velocity_mps']:.4f} m/s
                          {result['settling_velocity_mm_s']:.2f} mm/s
  Reynolds number:        {result['reynolds_number']:.2f}
  Flow regime:            {result['flow_regime']}
  Drag coefficient:       {result['drag_coefficient']:.3f}
""")
            pause()

        elif choice == 3:
            # Critical velocity
            print_subheader("CRITICAL DEPOSITION VELOCITY")
            pipe_dia = get_float("Pipe inner diameter (mm)", default=150, min_val=50)
            d50 = get_float("Particle d50 (mm)", default=0.5, min_val=0.01, max_val=100)
            sg = get_float("Particle SG", default=2.65, min_val=1.1, max_val=10)
            Cv = get_float("Volume concentration Cv (0-1 or %)", default=0.20, min_val=0, max_val=100)

            if Cv > 1:
                Cv = Cv / 100

            particle = ParticleParams(d50_mm=d50, particle_sg=sg)
            result = calc.calculate_critical_velocity(pipe_dia, particle, Cv)

            print_subheader("RESULTS")
            print(f"""
  Pipe diameter:          {result['pipe_diameter_mm']:.0f} mm
  Particle d50:           {result['particle_d50_mm']:.2f} mm
  Volume concentration:   {result['volume_concentration_Cv']*100:.1f}%

  Durand factor (F_L):    {result['durand_factor_FL']:.3f}

  Critical velocity (Durand):  {result['critical_velocity_durand_mps']:.2f} m/s
  Critical velocity (Wasp):    {result['critical_velocity_wasp_mps']:.2f} m/s

  RECOMMENDED MINIMUM:         {result['recommended_minimum_mps']:.2f} m/s
  (includes 20% safety margin)
""")
            pause()

        elif choice == 4:
            # Mass balance
            print_subheader("EXCAVATION MASS BALANCE")
            print("  Based on slurry circuit flow and density measurements")
            print()
            Q_in = get_float("Feed flow rate (m³/hr)", default=200, min_val=1)
            Q_out = get_float("Return flow rate (m³/hr)", default=205, min_val=1)
            rho_in = get_float("Feed density (kg/m³)", default=1080, min_val=1000, max_val=2000)
            rho_out = get_float("Return density (kg/m³)", default=1250, min_val=1000, max_val=2000)
            time_hr = get_float("Time period (hours)", default=1.0, min_val=0.1)

            circuit = SlurryCircuitParams(
                Q_in_m3_hr=Q_in, Q_out_m3_hr=Q_out,
                rho_in=rho_in, rho_out=rho_out
            )
            result = calc.calculate_excavation_mass_balance(circuit, time_hr)

            print_subheader("RESULTS")
            print(f"""
  Feed Cv:                {result['concentration_Cv_in']*100:.1f}%
  Return Cv:              {result['concentration_Cv_out']*100:.1f}%

  Excavated volume:       {result['excavated_volume_m3_hr']:.3f} m³/hr
  Excavated dry mass:     {result['excavated_dry_mass_kg_hr']:.1f} kg/hr

  TOTAL ({result['time_period_hours']:.1f} hours):
  ─────────────────────────────────────────────────────────────
  Total volume:           {result['total_excavated_volume_m3']:.3f} m³
  Total mass:             {result['total_excavated_mass_tonnes']:.3f} tonnes
""")
            pause()

        elif choice == 5:
            # Over-excavation check
            print_subheader("OVER-EXCAVATION CHECK")
            measured = get_float("Measured volume (m³)", min_val=0)
            theoretical = get_float("Theoretical volume (m³)", min_val=0.001)
            tolerance = get_float("Tolerance (%)", default=5.0, min_val=1, max_val=50)

            result = calc.check_overexcavation(measured, theoretical, tolerance)

            print_subheader("RESULTS")
            status_color = "" if result['status'] == "NORMAL" else "*** "
            print(f"""
  Measured volume:        {result['measured_volume_m3']:.3f} m³
  Theoretical volume:     {result['theoretical_volume_m3']:.3f} m³
  Difference:             {result['difference_m3']:.3f} m³ ({result['difference_percent']:.1f}%)
  Tolerance:              ±{result['tolerance_percent']:.0f}%

  {status_color}STATUS: {result['status']}
  {result['recommendation']}
""")
            pause()

        elif choice == 6:
            # Bentonite injection
            print_subheader("BENTONITE INJECTION REQUIREMENTS")
            pipe_od = get_float("Pipe outer diameter (mm)", default=1400, min_val=100)
            tunnel_length = get_float("Tunnel length (m)", default=500, min_val=10)
            print("  Ground types: clay, silt, sand, gravel, mixed, rock")
            ground = get_string("Ground type", default="sand",
                               options=["clay", "silt", "sand", "gravel", "mixed", "rock"])
            advance_rate = get_float("Advance rate (m/hr)", default=1.8, min_val=0.1)

            result = calc.calculate_bentonite_injection(pipe_od, tunnel_length, ground, advance_rate)

            print_subheader("RESULTS")
            print(f"""
  Ground type:            {result['ground_type']} ({result['ground_description']})
  Pipe diameter:          {result['pipe_diameter_mm']:.0f} mm
  Tunnel length:          {result['tunnel_length_m']:.0f} m

  INJECTION RATES:
  ─────────────────────────────────────────────────────────────
  Rate per meter:         {result['injection_rate_L_per_m']:.1f} L/m
  Total mix required:     {result['total_bentonite_mix_L']:.0f} L ({result['total_bentonite_mix_m3']:.1f} m³)

  MATERIALS (5% bentonite mix):
  ─────────────────────────────────────────────────────────────
  Bentonite powder:       {result['bentonite_powder_kg']:.0f} kg
  Water required:         {result['water_required_L']:.0f} L

  PUMPING (at {result['advance_rate_m_hr']:.1f} m/hr):
  ─────────────────────────────────────────────────────────────
  Pumping rate:           {result['pumping_rate_L_hr']:.1f} L/hr ({result['pumping_rate_L_min']:.2f} L/min)
""")
            pause()

        elif choice == 7:
            # Head loss
            print_subheader("SLURRY PIPELINE HEAD LOSS")
            pipe_dia = get_float("Pipe inner diameter (mm)", default=150, min_val=50)
            pipe_length = get_float("Pipe length (m)", default=200, min_val=10)
            flow_rate = get_float("Flow rate (m³/hr)", default=250, min_val=1)
            slurry_density = get_float("Slurry density (kg/m³)", default=1250, min_val=1000)
            d50 = get_float("Particle d50 (mm)", default=0.5, min_val=0.01)
            Cv = get_float("Volume concentration Cv (0-1 or %)", default=0.15, min_val=0, max_val=100)

            if Cv > 1:
                Cv = Cv / 100

            particle = ParticleParams(d50_mm=d50, particle_sg=2.65)
            result = calc.calculate_slurry_head_loss(
                pipe_dia, pipe_length, flow_rate, slurry_density, particle, Cv
            )

            print_subheader("RESULTS")
            print(f"""
  Velocity:               {result['velocity_mps']:.2f} m/s
  Reynolds number:        {result['reynolds_number']:.0f}

  HEAD LOSS:
  ─────────────────────────────────────────────────────────────
  Water only:             {result['head_loss_water_m']:.2f} m
  Slurry correction (φ):  {result['slurry_correction_phi']:.2f}
  Slurry head loss:       {result['head_loss_slurry_m']:.2f} m
  Increase:               +{result['head_loss_increase_percent']:.0f}%

  Pressure loss:          {result['pressure_loss_bar']:.2f} bar
""")
            pause()


# =============================================================================
# PIPEJACKING CALCULATOR
# =============================================================================

def pipejacking_menu():
    """Pipejacking calculator menu"""
    calc = PipejackingCalculator()

    while True:
        print_menu("PIPEJACKING CALCULATOR", [
            "Jacking force & speed",
            "Slurry flow rate table",
            "Power requirements",
            "Cable sizing"
        ])

        choice = get_choice("Select option", range(5))

        if choice == 0:
            return

        elif choice == 1:
            # Jacking force
            print_subheader("JACKING FORCE & SPEED")
            num_cyl = get_int("Number of cylinders", default=4, min_val=1, max_val=32)
            cyl_dia = get_float("Cylinder diameter (mm)", default=285, min_val=50)
            oil_flow = get_float("Oil flow rate (L/min)", default=60, min_val=1)
            max_pressure = get_float("Maximum pressure (bar)", default=500, min_val=50)
            pipe_length = get_float("Jacking pipe length (m)", default=3.0, min_val=0.5)

            params = JackingStationParams(
                num_cylinders=num_cyl,
                cylinder_diameter_mm=cyl_dia,
                oil_flow_lpm=oil_flow,
                max_pressure_bar=max_pressure
            )
            result = calc.calculate_jacking_force(params)

            print_subheader("RESULTS")
            time_per_pipe = (pipe_length * 100) / result.max_speed_cm_per_min
            print(f"""
  Combined bore area:     {result.combined_area_cm2:.2f} cm²

  FORCE:
  ─────────────────────────────────────────────────────────────
  Total jacking force:    {result.total_force_tonnes:.2f} tonnes
                          {result.total_force_tonnes * 9.81:.0f} kN

  SPEED:
  ─────────────────────────────────────────────────────────────
  Maximum speed:          {result.max_speed_cm_per_min:.2f} cm/min
                          {result.max_speed_cm_per_min * 0.6:.2f} m/hr

  Time per {pipe_length}m pipe:      {time_per_pipe:.1f} minutes
""")
            pause()

        elif choice == 2:
            # Slurry flow table
            print_subheader("SLURRY FLOW RATE TABLE")
            pipe_dia = get_float("Slurry pipe diameter (mm)", default=150, min_val=50)

            slurry_params = SlurryParams(pipe_diameter_mm=pipe_dia)
            velocities = [2.5, 3.0, 3.25, 3.5, 3.75, 4.0, 4.5, 5.0]
            table = calc.generate_slurry_flow_table(slurry_params, velocities)

            print_subheader(f"FLOW RATES FOR {pipe_dia:.0f}mm PIPE")
            print(f"""
  {'Velocity':<12} {'Flow (m³/hr)':<15} {'Flow (L/min)':<15} {'Notes':<20}
  {'─'*65}""")

            for i, row in enumerate(table):
                notes = ""
                if row['velocity_mps'] == 3.0:
                    notes = "Sandy conditions"
                elif row['velocity_mps'] == 3.5:
                    notes = "Larger/irregular"
                elif row['velocity_mps'] == 5.0:
                    notes = "Maximum recommended"

                print(f"  {row['velocity_mps']:<12.2f} {row['flow_rate_m3_per_hr']:<15.2f} {row['flow_rate_l_per_min']:<15.1f} {notes:<20}")

            pause()

        elif choice == 3:
            # Power requirements
            print_subheader("POWER REQUIREMENTS")
            print("  Enter motor powers (kW), enter 0 when done")
            print()

            motors = []
            motor_names = []
            while len(motors) < 10:
                name = input(f"  Motor {len(motors)+1} name (or press Enter to finish): ").strip()
                if not name:
                    break
                power = get_float(f"  Motor {len(motors)+1} power (kW)", default=0, min_val=0)
                motors.append(power)
                motor_names.append(name)

            if not motors:
                print("  No motors entered")
                pause()
                continue

            pf = get_float("Power factor", default=0.8, min_val=0.5, max_val=1.0)
            voltage = get_float("Voltage (V)", default=400, min_val=200, max_val=1000)

            power_params = PowerParams(power_factor=pf, voltage=voltage)
            result = calc.calculate_power_requirements(motors, power_params)

            print_subheader("RESULTS")
            print(f"""
  MOTORS:
  ─────────────────────────────────────────────────────────────""")
            for name, power in zip(motor_names, motors):
                print(f"  {name:<25} {power:>8.1f} kW")
            print(f"""
  ─────────────────────────────────────────────────────────────
  Total power:            {result['total_kw']:.1f} kW
  Power factor:           {result['power_factor']}
  Total apparent power:   {result['total_kva']:.1f} kVA

  GENERATOR SIZING (80%):
  ─────────────────────────────────────────────────────────────
  Recommended generator:  {result['generator_size_kva']:.0f} kVA
""")
            pause()

        elif choice == 4:
            # Cable sizing
            print_subheader("CABLE SIZING")
            kva = get_float("Load (kVA)", min_val=1)
            voltage = get_float("Voltage (V)", default=400, min_val=200, max_val=1000)
            three_phase = get_string("Three phase? (y/n)", default="y", options=["y", "n"]) == "y"

            amps = calc.calculate_cable_amperage(kva, voltage, three_phase)
            cable = calc.get_cable_size(amps)

            phase_str = "3-phase" if three_phase else "1-phase"
            print_subheader("RESULTS")
            print(f"""
  Load:                   {kva:.1f} kVA ({phase_str})
  Voltage:                {voltage:.0f} V

  Calculated amperage:    {amps:.1f} A

  CABLE RECOMMENDATION:
  ─────────────────────────────────────────────────────────────
  Cable size:             {cable['recommended_size_mm2']} mm²
  Cable rating:           {cable['cable_rating_amps']} A
  Safety margin:          {cable['safety_margin']:.1f} A

  (Based on BS5467 & BS7211, Table 4E2A)
""")
            pause()


# =============================================================================
# ADVANCED PIPEJACKING CALCULATIONS
# =============================================================================

def advanced_pipejacking_menu():
    """Advanced pipejacking calculations from exercise examples"""
    calc = PipejackingCalculator()

    while True:
        print_menu("ADVANCED PIPEJACKING CALCULATIONS", [
            "Overcut volume & bentonite flow",
            "Lubrication station requirements",
            "Excavation volume with swell factor",
            "Buoyancy check",
            "Groundwater force on face",
            "Specific pipe friction analysis",
            "Advance time with expanders",
            "Settlement from overcut",
            "Curve radius from joint gaps",
            "Penetration rate (cm/rotation)",
            "Slope/grade calculation",
            "Ground content from density"
        ])

        choice = get_choice("Select option", range(13))

        if choice == 0:
            return

        elif choice == 1:
            # Overcut volume
            print_subheader("OVERCUT VOLUME & BENTONITE FLOW")
            cutter_dia = get_float("Cutterhead diameter with overcut (mm)", default=2560)
            pipe_od = get_float("Pipe outer diameter (mm)", default=2500)
            advance_speed = get_float("Advance speed (cm/min)", default=10)
            loss_pct = get_float("Ground loss factor (%)", default=15, min_val=0, max_val=50)

            overcut = calc.calculate_overcut_volume(cutter_dia, pipe_od)
            bento_flow = calc.calculate_bentonite_flow_for_overcut(
                overcut['volume_per_m_m3'], advance_speed, 1 + loss_pct/100
            )

            print_subheader("RESULTS")
            print(f"""
  OVERCUT GEOMETRY:
  ─────────────────────────────────────────────────────────────
  Cutterhead diameter:    {cutter_dia:.0f} mm
  Pipe outer diameter:    {pipe_od:.0f} mm
  Overcut per side:       {overcut['overcut_per_side_mm']:.1f} mm

  Overcut area:           {overcut['overcut_area_m2']:.4f} m²
  Volume per meter:       {overcut['volume_per_m_m3']:.4f} m³/lfm
                          {overcut['volume_per_m_L']:.1f} L/lfm

  BENTONITE FLOW REQUIREMENT:
  ─────────────────────────────────────────────────────────────
  Advance speed:          {advance_speed:.1f} cm/min
  Time per meter:         {bento_flow['time_per_meter_min']:.1f} min

  Base flow (no loss):    {bento_flow['base_flow_L_min']:.1f} L/min
  With {loss_pct:.0f}% loss:          {bento_flow['total_flow_L_min']:.1f} L/min
                          {bento_flow['total_flow_m3_min']*60:.2f} m³/hr
""")
            pause()

        elif choice == 2:
            # Lubrication stations
            print_subheader("LUBRICATION STATION REQUIREMENTS")
            total_dist = get_float("Total tunnel distance (m)", default=325)
            machine_len = get_float("Machine length (m)", default=10)
            pipes_between = get_int("Pipes between lubrication stations", default=5, min_val=1)
            pipe_len = get_float("Pipe length (m)", default=3.0)
            vol_per_station = get_float("Volume per station per pipe (L)", default=50)
            advance_speed = get_float("Advance speed (cm/min)", default=10)

            result = calc.calculate_lubrication_stations(
                total_dist, machine_len, pipes_between, pipe_len,
                vol_per_station, advance_speed
            )

            print_subheader("RESULTS")
            print(f"""
  CONFIGURATION:
  ─────────────────────────────────────────────────────────────
  Total distance:         {result['total_distance_m']:.0f} m
  Machine length:         {result['machine_length_m']:.0f} m
  Pipe section length:    {result['pipe_section_length_m']:.0f} m

  LUBRICATION STATIONS:
  ─────────────────────────────────────────────────────────────
  Station spacing:        {result['station_spacing_m']:.0f} m ({pipes_between} pipes)
  Number of stations:     {result['num_stations']}

  Volume per station:     {result['volume_per_station_L']:.0f} L
  Total per pipe advance: {result['total_volume_per_pipe_L']:.0f} L ({result['total_volume_per_pipe_m3']:.2f} m³)

  Lubrication flow rate:  {result['lubrication_flow_L_min']:.1f} L/min
""")
            pause()

        elif choice == 3:
            # Excavation volume with swell
            print_subheader("EXCAVATION VOLUME WITH SWELL FACTOR")
            cutter_dia = get_float("Cutterhead diameter (mm)", default=2560)
            pipe_len = get_float("Pipe length (m)", default=3.0)
            swell = get_float("Swell factor", default=1.2, min_val=1.0, max_val=2.0)
            sep_eff = get_float("Separation efficiency (%)", default=95, min_val=50, max_val=100) / 100
            material_density = get_float("Material density (t/m³)", default=1.9)

            exc_vol = calc.calculate_excavation_volume(cutter_dia, pipe_len, swell, sep_eff)
            ground_mass = calc.calculate_ground_mass(exc_vol['separated_volume_m3'], material_density)

            print_subheader("RESULTS")
            print(f"""
  EXCAVATION VOLUME PER PIPE:
  ─────────────────────────────────────────────────────────────
  Cutterhead diameter:    {cutter_dia:.0f} mm
  Pipe length:            {pipe_len:.1f} m

  In-situ volume:         {exc_vol['insitu_volume_m3']:.2f} m³
  Swell factor:           {swell:.2f}
  Bulked volume:          {exc_vol['bulked_volume_m3']:.2f} m³

  Separation efficiency:  {sep_eff*100:.0f}%
  Separated volume:       {exc_vol['separated_volume_m3']:.2f} m³
  Lost to slurry:         {exc_vol['loss_volume_m3']:.2f} m³

  GROUND MASS:
  ─────────────────────────────────────────────────────────────
  Material density:       {material_density:.1f} t/m³
  Mass per pipe:          {ground_mass['mass_tonnes']:.2f} tonnes
""")
            pause()

        elif choice == 4:
            # Buoyancy check
            print_subheader("PIPE BUOYANCY CHECK")
            print("  Determines if pipe floats under groundwater conditions")
            print()
            pipe_od = get_float("Pipe outer diameter (mm)", default=2500)
            pipe_id = get_float("Pipe inner diameter (mm)", default=2000)
            pipe_len = get_float("Pipe length (m)", default=3.0)
            concrete_density = get_float("Concrete density (t/m³)", default=2.42)

            pipe = PipeParams(
                outer_diameter_mm=pipe_od,
                inner_diameter_mm=pipe_id,
                length_m=pipe_len,
                concrete_density=concrete_density
            )
            result = calc.calculate_buoyancy(pipe)

            print_subheader("RESULTS")
            print(f"""
  PIPE DIMENSIONS:
  ─────────────────────────────────────────────────────────────
  Outer diameter:         {pipe_od:.0f} mm
  Inner diameter:         {pipe_id:.0f} mm
  Wall thickness:         {(pipe_od-pipe_id)/2:.0f} mm
  Length:                 {pipe_len:.1f} m

  MASS CALCULATION:
  ─────────────────────────────────────────────────────────────
  Displaced water volume: {result['volume_displaced_m3']:.3f} m³
  Mass of displaced water:{result['mass_water_displaced_t']:.2f} tonnes

  Concrete volume:        {result['volume_concrete_m3']:.3f} m³
  Mass of pipe:           {result['mass_pipe_t']:.2f} tonnes

  BUOYANCY FORCE:         {result['buoyancy_force_t']:.2f} tonnes

  STATUS: {result['status']}
""")
            pause()

        elif choice == 5:
            # Groundwater force
            print_subheader("GROUNDWATER FORCE ON TUNNEL FACE")
            cutter_dia = get_float("Cutterhead diameter (mm)", default=2560)
            water_pressure = get_float("Water pressure at face (bar)", default=0.6)

            result = calc.calculate_groundwater_force(cutter_dia, water_pressure)

            print_subheader("RESULTS")
            print(f"""
  PARAMETERS:
  ─────────────────────────────────────────────────────────────
  Cutterhead diameter:    {cutter_dia:.0f} mm
  Face area:              {result['face_area_m2']:.2f} m²
  Water pressure:         {water_pressure:.2f} bar

  GROUNDWATER FORCE:
  ─────────────────────────────────────────────────────────────
  Force on face:          {result['force_tonnes']:.2f} tonnes
                          {result['force_kN']:.0f} kN

  NOTE: {result['note']}
""")
            pause()

        elif choice == 6:
            # Specific friction
            print_subheader("SPECIFIC PIPE FRICTION ANALYSIS")
            print("  Calculates friction in t/m² for pipe sections")
            print("  Reference: <0.5=Good, 0.5-1.0=Attention, >1.0=Critical")
            print()

            pipe_dia = get_float("Pipe outer diameter (mm)", default=2500)
            force = get_float("Jacking force for section (tonnes)", default=650)
            section_len = get_float("Section length (m)", default=100)

            result = calc.calculate_specific_friction(force, pipe_dia, section_len)

            print_subheader("RESULTS")
            print(f"""
  PARAMETERS:
  ─────────────────────────────────────────────────────────────
  Pipe diameter:          {pipe_dia:.0f} mm
  Section length:         {section_len:.0f} m
  Surface area:           {result['surface_area_m2']:.1f} m²

  Jacking force:          {force:.0f} tonnes

  SPECIFIC FRICTION:      {result['specific_friction_t_m2']:.2f} t/m²

  STATUS: {result['status']}
  {result['recommendation']}

  Reference values:
  ─────────────────────────────────────────────────────────────
  < 0.5 t/m²:   GOOD - Normal operation
  0.5-1.0 t/m²: ATTENTION - Monitor closely
  > 1.0 t/m²:   CRITICAL - Take action
""")
            pause()

        elif choice == 7:
            # Advance time
            print_subheader("ADVANCE TIME WITH EXPANDERS")
            pipe_len = get_float("Pipe length (m)", default=3.0)
            stroke_len = get_float("Cylinder stroke length (mm)", default=600)
            advance_speed = get_float("Advance speed (cm/min)", default=10)
            retract_speed = get_float("Retract speed (cm/min)", default=20)
            changeover_time = get_float("Changeover time (min)", default=1.0)
            num_sets = get_int("Number of cylinder sets (machine + expanders)", default=3, min_val=1)

            result = calc.calculate_advance_time(
                pipe_len, stroke_len, advance_speed, retract_speed,
                changeover_time, num_sets
            )

            print_subheader("RESULTS")
            print(f"""
  PARAMETERS:
  ─────────────────────────────────────────────────────────────
  Pipe length:            {pipe_len:.1f} m
  Stroke length:          {stroke_len:.0f} mm
  Number of cylinder sets:{num_sets}

  SPEEDS:
  ─────────────────────────────────────────────────────────────
  Advance speed:          {advance_speed:.0f} cm/min
  Retract speed:          {retract_speed:.0f} cm/min
  Changeover time:        {changeover_time:.1f} min

  TIME CALCULATION:
  ─────────────────────────────────────────────────────────────
  Strokes per pipe:       {result['strokes_per_pipe']}
  Time per stroke:        {result['time_per_stroke_min']:.1f} min

  TOTAL TIME PER PIPE:    {result['total_time_per_pipe_min']:.1f} minutes
  Pipes per hour:         {result['pipes_per_hour']:.2f}
""")
            pause()

        elif choice == 8:
            # Settlement
            print_subheader("SETTLEMENT FROM OVERCUT")
            cutter_dia = get_float("Cutterhead diameter (mm)", default=2560)
            pipe_od = get_float("Pipe outer diameter (mm)", default=2500)

            overcut = calc.calculate_overcut_volume(cutter_dia, pipe_od)
            settlement = calc.calculate_settlement_from_overcut(
                overcut['volume_per_m_m3'], cutter_dia
            )

            overcut_per_side = (cutter_dia - pipe_od) / 2

            print_subheader("RESULTS")
            print(f"""
  OVERCUT:
  ─────────────────────────────────────────────────────────────
  Cutterhead diameter:    {cutter_dia:.0f} mm
  Pipe outer diameter:    {pipe_od:.0f} mm
  Overcut per side:       {overcut_per_side:.0f} mm

  Overcut volume:         {overcut['volume_per_m_m3']:.4f} m³/m

  SETTLEMENT ESTIMATE:
  ─────────────────────────────────────────────────────────────
  Method 1 (pure overcut):
    Settlement =          {overcut_per_side/10:.1f} cm (worst case localized)

  Method 2 (distributed):
    Area above pipe:      {settlement['area_above_pipe_m2']:.2f} m²
    Settlement =          {settlement['settlement_distributed_cm']:.1f} cm (average)

  Note: {settlement['note']}
""")
            pause()

        elif choice == 9:
            # Curve radius from joints
            print_subheader("CURVE RADIUS FROM JOINT MEASUREMENTS")
            print("  Enter joint gap measurements at 4 positions")
            print()
            joint_peak = get_float("Joint gap at PEAK/crown (mm)", default=25)
            joint_right = get_float("Joint gap at RIGHT abutment (mm)", default=20)
            joint_invert = get_float("Joint gap at INVERT (mm)", default=25)
            joint_left = get_float("Joint gap at LEFT abutment (mm)", default=30)
            pipe_dia = get_float("Pipe outer diameter (mm)", default=2560)
            pipe_len = get_float("Pipe length (m)", default=3.0)

            result = calc.calculate_curve_radius_from_joints(
                joint_peak, joint_right, joint_invert, joint_left,
                pipe_dia, pipe_len
            )

            print_subheader("RESULTS")
            radius_str = f"{result['curve_radius_m']:.0f} m" if result['curve_radius_m'] < 10000 else "STRAIGHT"
            print(f"""
  JOINT MEASUREMENTS:
  ─────────────────────────────────────────────────────────────
  Peak (12 o'clock):      {joint_peak:.0f} mm
  Right (3 o'clock):      {joint_right:.0f} mm
  Invert (6 o'clock):     {joint_invert:.0f} mm
  Left (9 o'clock):       {joint_left:.0f} mm

  GAP ANALYSIS:
  ─────────────────────────────────────────────────────────────
  Horizontal gap (L-R):   {result['horizontal_gap_mm']:.0f} mm → Curve {result['horizontal_direction']}
  Vertical gap (P-I):     {result['vertical_gap_mm']:.0f} mm → Bend {result['vertical_direction']}

  PIPE ANGLE:
  ─────────────────────────────────────────────────────────────
  Horizontal angle:       {result['horizontal_angle_deg']:.3f}°
  Vertical angle:         {result['vertical_angle_deg']:.3f}°

  CURVE RADIUS:           {radius_str}
""")
            pause()

        elif choice == 10:
            # Penetration rate
            print_subheader("PENETRATION RATE (cm/rotation)")
            advance_speed = get_float("Advance speed (cm/min)", default=10)
            cutter_rpm = get_float("Cutterhead RPM", default=4.9)

            result = calc.calculate_penetration_rate(advance_speed, cutter_rpm)

            print_subheader("RESULTS")
            print(f"""
  PARAMETERS:
  ─────────────────────────────────────────────────────────────
  Advance speed:          {advance_speed:.1f} cm/min
  Cutterhead RPM:         {cutter_rpm:.1f}

  PENETRATION RATE:
  ─────────────────────────────────────────────────────────────
  Penetration:            {result['penetration_cm_per_rev']:.2f} cm/revolution
                          {result['penetration_mm_per_rev']:.1f} mm/revolution

  Note: {result['note']}
""")
            pause()

        elif choice == 11:
            # Slope calculation
            print_subheader("SLOPE / GRADE CALCULATION")
            distance = get_float("Horizontal distance (m)", default=465)
            slope_pct = get_float("Slope (%)", default=1.2)

            result = calc.calculate_slope(distance, slope_pct)

            print_subheader("RESULTS")
            print(f"""
  PARAMETERS:
  ─────────────────────────────────────────────────────────────
  Distance:               {distance:.0f} m
  Slope:                  {slope_pct:.2f}%
                          {result['slope_permille']:.1f}‰ (mm/m)
                          {result['slope_ratio']}

  HEIGHT DIFFERENCE:
  ─────────────────────────────────────────────────────────────
  Direction:              {result['direction'].upper()}
  Height difference:      {result['height_difference_m']:.2f} m
                          {result['height_difference_cm']:.0f} cm
                          {result['height_difference_mm']:.0f} mm
""")
            pause()

        elif choice == 12:
            # Ground from density
            print_subheader("GROUND CONTENT FROM SLURRY DENSITY")
            print("  Method 1: Single density measurement")
            slurry_density = get_float("Slurry density (kg/L)", default=1.10)
            base_density = get_float("Base fluid density (kg/L)", default=1.0)

            result1 = calc.calculate_ground_content_from_density(slurry_density, base_density)

            print(f"""
  SINGLE MEASUREMENT:
  ─────────────────────────────────────────────────────────────
  Slurry density:         {slurry_density:.2f} kg/L
  Base fluid density:     {base_density:.2f} kg/L
  Ground content:         {result1['ground_content_kg_m3']:.0f} kg/m³
""")

            print("  Method 2: Density change during pipe advance")
            initial_density = get_float("Initial density (kg/L)", default=1.05)
            final_density = get_float("Final density (kg/L)", default=1.08)
            slurry_volume = get_float("Slurry system volume (m³)", default=50)

            result2 = calc.calculate_ground_from_density_change(
                initial_density, final_density, slurry_volume
            )

            print_subheader("RESULTS")
            print(f"""
  DENSITY CHANGE DURING ADVANCE:
  ─────────────────────────────────────────────────────────────
  Initial density:        {initial_density:.2f} kg/L
  Final density:          {final_density:.2f} kg/L
  Density change:         {result2['density_change_kg_L']:.3f} kg/L

  Ground content:         {result2['ground_content_kg_m3']:.0f} kg/m³
  System volume:          {slurry_volume:.0f} m³

  TOTAL GROUND ENTERED:   {result2['total_ground_entered_kg']:.0f} kg
""")
            pause()


# =============================================================================
# QUICK CALCULATIONS
# =============================================================================

def quick_calc_menu():
    """Quick calculation shortcuts"""
    while True:
        print_menu("QUICK CALCULATIONS", [
            "Slurry mixture SG from concentration",
            "Pipe velocity from flow rate",
            "Jacking force from pressure",
            "kW to kVA conversion",
            "Excavation volume from advance rate"
        ])

        choice = get_choice("Select option", range(6))

        if choice == 0:
            return

        elif choice == 1:
            # Mixture SG
            print_subheader("Mixture SG from Concentration")
            S = get_float("Solids SG", default=2.65)
            Sw = get_float("Liquid SG", default=1.0)
            Cv = get_float("Volume concentration (%)", default=25, min_val=0, max_val=100) / 100

            Sm = Sw + Cv * (S - Sw)
            Cw = S * Cv / Sm

            print(f"\n  Mixture SG (Sm):      {Sm:.3f}")
            print(f"  Weight concentration: {Cw*100:.1f}%")
            pause()

        elif choice == 2:
            # Velocity
            print_subheader("Pipe Velocity from Flow Rate")
            pipe_dia = get_float("Pipe diameter (mm)", default=150)
            flow = get_float("Flow rate (m³/hr)", default=200)

            area = math.pi * (pipe_dia/2000)**2
            velocity = (flow/3600) / area

            print(f"\n  Pipe area:            {area:.6f} m²")
            print(f"  Velocity:             {velocity:.2f} m/s")

            if velocity < 3.0:
                print("  WARNING: Below minimum transport velocity!")
            elif velocity > 5.0:
                print("  WARNING: Above maximum recommended velocity!")
            pause()

        elif choice == 3:
            # Jacking force
            print_subheader("Jacking Force from Pressure")
            num_cyl = get_int("Number of cylinders", default=4)
            dia = get_float("Cylinder diameter (mm)", default=285)
            pressure = get_float("Pressure (bar)", default=500)

            area = math.pi * (dia/20)**2 * num_cyl  # cm²
            force = pressure * area / 1000  # tonnes

            print(f"\n  Combined area:        {area:.2f} cm²")
            print(f"  Total force:          {force:.2f} tonnes")
            print(f"                        {force*9.81:.0f} kN")
            pause()

        elif choice == 4:
            # kW to kVA
            print_subheader("kW to kVA Conversion")
            kw = get_float("Power (kW)", min_val=0)
            pf = get_float("Power factor", default=0.8, min_val=0.5, max_val=1.0)

            kva = kw / pf

            print(f"\n  Apparent power:       {kva:.2f} kVA")
            print(f"  At 400V, 3-phase:     {kva*1000/(400*1.732):.1f} A")
            pause()

        elif choice == 5:
            # Excavation volume
            print_subheader("Excavation Volume from Advance Rate")
            cutter_dia = get_float("Cutter diameter (mm)", default=1500)
            advance = get_float("Advance rate (mm/min)", default=30)

            area = math.pi * (cutter_dia/2000)**2
            vol_hr = area * (advance/1000) * 60

            print(f"\n  Cutter area:          {area:.4f} m²")
            print(f"  Advance rate:         {advance*60/1000:.2f} m/hr")
            print(f"  Excavation rate:      {vol_hr:.3f} m³/hr")
            pause()


# =============================================================================
# MAIN MENU
# =============================================================================

def main_menu():
    """Main application menu"""
    while True:
        clear_screen()
        print_header("MTBM UNIFIED CALCULATOR", 70)
        print("""
  Microtunneling & Pipejacking Calculation Suite
  ─────────────────────────────────────────────────────────────
  Based on:
  - Excellence Pump Industry Slurry Manual
  - Hitchhiker's Guide to Pipejacking (Lutz Henke)
  - Industry research papers (Duhme et al., Durand-Condolios)
""")
        print_menu("MAIN MENU", [
            "Slurry Properties Calculator (S, Sw, Sm, Cw, Cv)",
            "Slurry Flow Calculator (excavation, velocity, head loss)",
            "Pipejacking Calculator (force, speed, power, cables)",
            "Advanced Pipejacking (overcut, friction, settlement, curves)",
            "Quick Calculations"
        ], 70)

        choice = get_choice("Select module", range(6))

        if choice == 0:
            print("\n  Goodbye!\n")
            sys.exit(0)
        elif choice == 1:
            slurry_properties_menu()
        elif choice == 2:
            slurry_flow_menu()
        elif choice == 3:
            pipejacking_menu()
        elif choice == 4:
            advanced_pipejacking_menu()
        elif choice == 5:
            quick_calc_menu()


# =============================================================================
# COMMAND LINE INTERFACE
# =============================================================================

def show_help():
    """Show command line help"""
    print("""
MTBM Unified Calculator CLI
===========================

Usage:
  python mtbm_calculator_cli.py              Interactive mode (default)
  python mtbm_calculator_cli.py --quick      Quick calculation mode
  python mtbm_calculator_cli.py --help       Show this help

  Command-line calculations (non-interactive):
  python mtbm_calculator_cli.py --calc <type> <args>

Modules:
  1. Slurry Properties  - Calculate S, Sw, Sm, Cw, Cv relationships
  2. Slurry Flow        - Excavation, critical velocity, mass balance
  3. Pipejacking        - Force, speed, power, cable sizing
  4. Quick Calculations - Common one-off calculations

Command-line Calculation Types (--calc):
  mixture-sg <S> <Sw> <Cv>      Calculate mixture SG from concentration
  velocity <pipe_mm> <flow_m3h> Calculate velocity from flow rate
  force <n_cyl> <dia_mm> <bar>  Calculate jacking force
  settling <d50_mm> [sg]        Calculate settling velocity
  critical <pipe_mm> <d50> <Cv> Calculate critical velocity
  excavation <dia_mm> <mm_min>  Calculate excavation volume rate
  kva <kw> [pf]                 Convert kW to kVA

Examples:
  # Interactive mode
  python mtbm_calculator_cli.py

  # Calculate mixture SG for 30% sand in water
  python mtbm_calculator_cli.py --calc mixture-sg 2.65 1.0 0.30

  # Calculate velocity in 150mm pipe at 200 m³/hr
  python mtbm_calculator_cli.py --calc velocity 150 200

  # Calculate jacking force with 4x285mm cylinders at 500 bar
  python mtbm_calculator_cli.py --calc force 4 285 500

  # Calculate settling velocity for 0.5mm sand
  python mtbm_calculator_cli.py --calc settling 0.5

  # Calculate critical velocity in 150mm pipe, 0.5mm sand, 20% Cv
  python mtbm_calculator_cli.py --calc critical 150 0.5 0.20
""")


def run_cmdline_calc(calc_type: str, args: list):
    """Run a command-line calculation"""
    flow_calc = SlurryFlowCalculator()
    pipe_calc = PipejackingCalculator()

    try:
        if calc_type == 'mixture-sg':
            if len(args) < 3:
                print("Usage: --calc mixture-sg <S> <Sw> <Cv>")
                return
            S, Sw, Cv = float(args[0]), float(args[1]), float(args[2])
            if Cv > 1:
                Cv = Cv / 100
            Sm = Sw + Cv * (S - Sw)
            Cw = S * Cv / Sm
            print(f"Mixture SG (Sm): {Sm:.4f}")
            print(f"Weight conc (Cw): {Cw*100:.2f}%")

        elif calc_type == 'velocity':
            if len(args) < 2:
                print("Usage: --calc velocity <pipe_mm> <flow_m3h>")
                return
            pipe_dia, flow = float(args[0]), float(args[1])
            area = math.pi * (pipe_dia/2000)**2
            velocity = (flow/3600) / area
            print(f"Pipe area: {area:.6f} m²")
            print(f"Velocity: {velocity:.3f} m/s")
            if velocity < 3.0:
                print("WARNING: Below minimum transport velocity (3.0 m/s)")
            elif velocity > 5.0:
                print("WARNING: Above maximum recommended (5.0 m/s)")

        elif calc_type == 'force':
            if len(args) < 3:
                print("Usage: --calc force <n_cyl> <dia_mm> <pressure_bar>")
                return
            n_cyl, dia, pressure = int(args[0]), float(args[1]), float(args[2])
            area = math.pi * (dia/20)**2 * n_cyl
            force = pressure * area / 1000
            print(f"Combined area: {area:.2f} cm²")
            print(f"Total force: {force:.2f} tonnes ({force*9.81:.0f} kN)")

        elif calc_type == 'settling':
            if len(args) < 1:
                print("Usage: --calc settling <d50_mm> [sg]")
                return
            d50 = float(args[0])
            sg = float(args[1]) if len(args) > 1 else 2.65
            particle = ParticleParams(d50_mm=d50, particle_sg=sg)
            result = flow_calc.calculate_settling_velocity(particle)
            print(f"Particle: {d50}mm, SG={sg}")
            print(f"Settling velocity: {result['settling_velocity_mps']:.4f} m/s ({result['settling_velocity_mm_s']:.2f} mm/s)")
            print(f"Flow regime: {result['flow_regime']}")

        elif calc_type == 'critical':
            if len(args) < 3:
                print("Usage: --calc critical <pipe_mm> <d50_mm> <Cv>")
                return
            pipe_dia, d50, Cv = float(args[0]), float(args[1]), float(args[2])
            if Cv > 1:
                Cv = Cv / 100
            particle = ParticleParams(d50_mm=d50, particle_sg=2.65)
            result = flow_calc.calculate_critical_velocity(pipe_dia, particle, Cv)
            print(f"Pipe: {pipe_dia}mm, Particle: {d50}mm, Cv: {Cv*100}%")
            print(f"Critical velocity (Durand): {result['critical_velocity_durand_mps']:.2f} m/s")
            print(f"Critical velocity (Wasp): {result['critical_velocity_wasp_mps']:.2f} m/s")
            print(f"Recommended minimum: {result['recommended_minimum_mps']:.2f} m/s")

        elif calc_type == 'excavation':
            if len(args) < 2:
                print("Usage: --calc excavation <cutter_dia_mm> <advance_mm_min>")
                return
            cutter_dia, advance = float(args[0]), float(args[1])
            area = math.pi * (cutter_dia/2000)**2
            vol_hr = area * (advance/1000) * 60
            print(f"Cutter diameter: {cutter_dia}mm")
            print(f"Cutter area: {area:.4f} m²")
            print(f"Advance rate: {advance}mm/min ({advance*60/1000:.2f} m/hr)")
            print(f"Excavation rate: {vol_hr:.3f} m³/hr")

        elif calc_type == 'kva':
            if len(args) < 1:
                print("Usage: --calc kva <kw> [pf]")
                return
            kw = float(args[0])
            pf = float(args[1]) if len(args) > 1 else 0.8
            kva = kw / pf
            amps = kva * 1000 / (400 * 1.732)
            print(f"Power: {kw} kW at PF={pf}")
            print(f"Apparent power: {kva:.2f} kVA")
            print(f"Current (400V, 3-ph): {amps:.1f} A")

        else:
            print(f"Unknown calculation type: {calc_type}")
            print("Run with --help to see available calculations")

    except (ValueError, IndexError) as e:
        print(f"Error: {e}")
        print("Run with --help to see usage")


def main():
    """Main entry point"""
    if len(sys.argv) > 1:
        arg = sys.argv[1].lower()
        if arg in ['--help', '-h', 'help']:
            show_help()
            return
        elif arg in ['--quick', '-q', 'quick']:
            quick_calc_menu()
            return
        elif arg in ['--calc', '-c', 'calc']:
            if len(sys.argv) < 3:
                print("Usage: --calc <type> <args>")
                print("Run with --help to see available calculations")
                return
            calc_type = sys.argv[2].lower()
            calc_args = sys.argv[3:]
            run_cmdline_calc(calc_type, calc_args)
            return

    # Default: interactive mode
    main_menu()


if __name__ == "__main__":
    main()
