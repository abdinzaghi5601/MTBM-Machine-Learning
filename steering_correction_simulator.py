"""
Steering Correction Simulator
==============================

Demonstrates how to progressively correct machine deviations to zero
over multiple pipe installations.

This script shows:
1. How formulas work step-by-step
2. How to plan corrections over time
3. How to bring horizontal and vertical deviations to zero
"""

from typing import Optional
from steering_calculator import (
    MachineParameters,
    SteeringCommand,
    CylinderReadings,
    SteeringCalculator,
    GroundCondition
)
import math


def simulate_correction_to_zero(
    initial_pitch: float,
    initial_yaw: float,
    target_pitch: float = 0.0,
    target_yaw: float = 0.0,
    max_pipes: int = 5,
    correction_rate: float = 0.5,  # Correct 50% of deviation per pipe
    ground_condition: Optional[GroundCondition] = None
):
    """
    Simulate progressive correction to bring pitch/yaw to zero
    
    Args:
        initial_pitch: Starting pitch (mm/m)
        initial_yaw: Starting yaw (mm/m)
        target_pitch: Target pitch (default: 0.0)
        target_yaw: Target yaw (default: 0.0)
        max_pipes: Maximum number of pipes to simulate
        correction_rate: Fraction of deviation to correct per pipe (0.0-1.0)
    
    Returns:
        List of correction steps with detailed information
    """
    
    # Setup calculator
    params = MachineParameters(
        num_cylinders=3,
        stroke=50.0,
        mounting_diameter=715.0,
        pipe_length=3000.0,
        vertical_angle=1.49
    )
    calc = SteeringCalculator(params)
    
    steps = []
    current_pitch = initial_pitch
    current_yaw = initial_yaw
    pipe_number = 0
    
    print("="*80)
    print("STEERING CORRECTION SIMULATION: Bringing Deviations to Zero")
    print("="*80)
    print(f"\nInitial State:")
    print(f"  Pitch: {initial_pitch:+7.2f} mm/m")
    print(f"  Yaw:   {initial_yaw:+7.2f} mm/m")
    print(f"\nTarget State:")
    print(f"  Pitch: {target_pitch:+7.2f} mm/m")
    print(f"  Yaw:   {target_yaw:+7.2f} mm/m")
    print(f"\nCorrection Strategy: {correction_rate*100:.0f}% of deviation per pipe")
    
    if ground_condition:
        max_rate = ground_condition.get_max_steering_rate()
        recommended_max = ground_condition.get_recommended_max()
        print(f"\nGround Condition: {ground_condition.value.upper()}")
        print(f"  Maximum allowed: {max_rate} mm/m")
        print(f"  Recommended max: {recommended_max} mm/m")
        print(f"  ⚠️  Corrections will be limited to comply with ground condition constraints")
    
    print("="*80)
    
    while pipe_number < max_pipes:
        pipe_number += 1
        
        # Calculate remaining deviation
        pitch_deviation = current_pitch - target_pitch
        yaw_deviation = current_yaw - target_yaw
        
        # Check if we're close enough to target
        if abs(pitch_deviation) < 0.1 and abs(yaw_deviation) < 0.1:
            print(f"\n✅ Target achieved after {pipe_number-1} pipes!")
            print(f"   Final Pitch: {current_pitch:+7.2f} mm/m")
            print(f"   Final Yaw:   {current_yaw:+7.2f} mm/m")
            break
        
        # Calculate required correction (proportional to remaining deviation)
        pitch_correction = -pitch_deviation * correction_rate
        yaw_correction = -yaw_deviation * correction_rate
        
        # Plan correction
        correction_plan = calc.plan_correction(
            current_pitch=current_pitch,
            current_yaw=current_yaw,
            target_pitch=current_pitch + pitch_correction,
            target_yaw=current_yaw + yaw_correction,
            ground_condition=ground_condition
        )
        
        # Calculate expected result
        expected_pitch = current_pitch + pitch_correction
        expected_yaw = current_yaw + yaw_correction
        
        # Store step information
        step_info = {
            'pipe_number': pipe_number,
            'current_pitch': current_pitch,
            'current_yaw': current_yaw,
            'pitch_correction': pitch_correction,
            'yaw_correction': yaw_correction,
            'cylinder_positions': correction_plan['cylinder_positions'],
            'expected_pitch': expected_pitch,
            'expected_yaw': expected_yaw,
            'feasible': correction_plan['feasibility']['is_feasible'],
            'correction_per_pipe': correction_plan['correction_per_pipe']
        }
        steps.append(step_info)
        
        # Display step
        print(f"\n{'='*80}")
        print(f"PIPE {pipe_number}")
        print(f"{'='*80}")
        print(f"\nCurrent State:")
        print(f"  Pitch: {current_pitch:+7.2f} mm/m  (deviation from target: {pitch_deviation:+7.2f} mm/m)")
        print(f"  Yaw:   {current_yaw:+7.2f} mm/m  (deviation from target: {yaw_deviation:+7.2f} mm/m)")
        
        # Get actual correction (may have been limited by ground condition)
        actual_pitch_corr = correction_plan['required_correction']['pitch']
        actual_yaw_corr = correction_plan['required_correction']['yaw']
        
        print(f"\nRequired Correction:")
        print(f"  Pitch Correction: {actual_pitch_corr:+7.2f} mm/m")
        print(f"  Yaw Correction:   {actual_yaw_corr:+7.2f} mm/m")
        
        # Show ground condition validation if present
        if correction_plan.get('ground_condition_validation'):
            gc_val = correction_plan['ground_condition_validation']
            total_rate = math.sqrt(actual_pitch_corr**2 + actual_yaw_corr**2)
            print(f"\nGround Condition Check:")
            print(f"  Total steering rate: {total_rate:.2f} mm/m")
            if gc_val['was_limited']:
                print(f"  ⚠️  LIMITED from {gc_val['original_rate']:.2f} mm/m to {total_rate:.2f} mm/m")
                print(f"     (Original: Pitch={pitch_correction:+.2f}, Yaw={yaw_correction:+.2f})")
            else:
                print(f"  ✅ Within limits for {gc_val['ground_condition']} ground")
        
        print(f"\nCylinder Positions:")
        for cyl, pos in correction_plan['cylinder_positions'].items():
            status = calc._get_position_status(pos)
            print(f"  {cyl:12s}: {pos:6.2f} mm  [{status}]")
        
        print(f"\nExpected Result After This Pipe:")
        print(f"  Pitch: {expected_pitch:+7.2f} mm/m")
        print(f"  Yaw:   {expected_yaw:+7.2f} mm/m")
        
        print(f"\nCorrection Per Pipe (in mm):")
        print(f"  Pitch: {correction_plan['correction_per_pipe']['pitch_per_pipe']:+7.2f} mm")
        print(f"  Yaw:   {correction_plan['correction_per_pipe']['yaw_per_pipe']:+7.2f} mm")
        
        if not correction_plan['feasibility']['is_feasible']:
            print(f"\n⚠️  WARNING: {correction_plan['feasibility']['reason']}")
        
        if correction_plan['warnings']:
            print(f"\n⚠️  Warnings:")
            for warning in correction_plan['warnings']:
                print(f"    • {warning}")
        
        # Update for next iteration (use actual correction applied)
        current_pitch = correction_plan['expected_result']['pitch_after_pipe']
        current_yaw = correction_plan['expected_result']['yaw_after_pipe']
    
    print(f"\n{'='*80}")
    print("SIMULATION COMPLETE")
    print(f"{'='*80}")
    
    return steps


def demonstrate_formulas():
    """
    Demonstrate how the formulas work with detailed calculations
    """
    
    print("\n" + "="*80)
    print("FORMULA DEMONSTRATION: How Cylinders Control Steering")
    print("="*80)
    
    params = MachineParameters(
        num_cylinders=3,
        stroke=50.0,
        mounting_diameter=715.0,
        pipe_length=3000.0
    )
    
    calc = SteeringCalculator(params)
    
    # Example: Correct pitch = -10 mm/m, yaw = +15 mm/m
    pitch_correction = -10.0
    yaw_correction = +15.0
    
    print(f"\nExample Correction:")
    print(f"  Pitch Correction: {pitch_correction} mm/m")
    print(f"  Yaw Correction:   {yaw_correction} mm/m")
    
    print(f"\nMachine Parameters:")
    print(f"  Mounting Diameter: {params.mounting_diameter} mm")
    print(f"  Mounting Radius:   {params.mounting_diameter/2} mm = {(params.mounting_diameter/2)/1000:.3f} m")
    print(f"  Stroke Center:     {params.stroke/2} mm")
    
    # Calculate manually to show formulas
    radius_m = (params.mounting_diameter / 2) / 1000
    stroke_center = params.stroke / 2
    
    pitch_effect = pitch_correction * radius_m
    yaw_effect = yaw_correction * radius_m
    
    print(f"\n{'='*80}")
    print("STEP-BY-STEP CALCULATION")
    print(f"{'='*80}")
    print(f"\n1. Calculate Effects:")
    print(f"   Pitch Effect = {pitch_correction} mm/m × {radius_m:.3f} m = {pitch_effect:.3f} mm")
    print(f"   Yaw Effect   = {yaw_correction} mm/m × {radius_m:.3f} m = {yaw_effect:.3f} mm")
    
    print(f"\n2. Calculate Cylinder Positions:")
    print(f"   Stroke Center = {stroke_center} mm")
    
    # Cylinder 1 (0° - Top)
    cyl1 = stroke_center + pitch_effect
    print(f"\n   Cylinder 1 (Top, 0°):")
    print(f"     = {stroke_center} + {pitch_effect:.3f}")
    print(f"     = {cyl1:.2f} mm")
    
    # Cylinder 2 (120°)
    cos120 = math.cos(math.radians(120))
    sin120 = math.sin(math.radians(120))
    cyl2 = stroke_center + (pitch_effect * cos120) + (yaw_effect * sin120)
    print(f"\n   Cylinder 2 (120°):")
    print(f"     = {stroke_center} + ({pitch_effect:.3f} × {cos120:.3f}) + ({yaw_effect:.3f} × {sin120:.3f})")
    print(f"     = {stroke_center} + {pitch_effect * cos120:.3f} + {yaw_effect * sin120:.3f}")
    print(f"     = {cyl2:.2f} mm")
    
    # Cylinder 3 (240°)
    cos240 = math.cos(math.radians(240))
    sin240 = math.sin(math.radians(240))
    cyl3 = stroke_center + (pitch_effect * cos240) + (yaw_effect * sin240)
    print(f"\n   Cylinder 3 (240°):")
    print(f"     = {stroke_center} + ({pitch_effect:.3f} × {cos240:.3f}) + ({yaw_effect:.3f} × {sin240:.3f})")
    print(f"     = {stroke_center} + {pitch_effect * cos240:.3f} + {yaw_effect * sin240:.3f}")
    print(f"     = {cyl3:.2f} mm")
    
    # Verify with calculator
    print(f"\n3. Verification (using calculator):")
    steering = SteeringCommand(pitch=pitch_correction, yaw=yaw_correction)
    cylinders = calc.calculate_cylinders(steering)
    print(f"   Cylinder 1: {cylinders['cylinder_1']:.2f} mm")
    print(f"   Cylinder 2: {cylinders['cylinder_2']:.2f} mm")
    print(f"   Cylinder 3: {cylinders['cylinder_3']:.2f} mm")
    
    # Reverse calculation
    print(f"\n4. Reverse Calculation (Cylinders → Pitch/Yaw):")
    readings = CylinderReadings(
        cylinder_1=cylinders['cylinder_1'],
        cylinder_2=cylinders['cylinder_2'],
        cylinder_3=cylinders['cylinder_3']
    )
    reverse_steering = calc.calculate_steering(readings)
    print(f"   From cylinder positions, we get:")
    print(f"   Pitch: {reverse_steering.pitch:.2f} mm/m")
    print(f"   Yaw:   {reverse_steering.yaw:.2f} mm/m")
    print(f"   (Should match input: Pitch={pitch_correction}, Yaw={yaw_correction})")


def show_correction_timeline():
    """
    Show how corrections accumulate over multiple pipes
    """
    
    print("\n" + "="*80)
    print("CORRECTION TIMELINE: How Deviations Reduce Over Time")
    print("="*80)
    
    # Example: Starting with significant deviations
    initial_pitch = 15.0
    initial_yaw = -20.0
    
    print(f"\nStarting Deviations:")
    print(f"  Pitch: {initial_pitch:+7.2f} mm/m")
    print(f"  Yaw:   {initial_yaw:+7.2f} mm/m")
    
    params = MachineParameters(pipe_length=3000.0)
    pipe_length_m = params.pipe_length / 1000  # 3 meters
    
    print(f"\nPipe Length: {params.pipe_length} mm = {pipe_length_m} meters")
    print(f"\n{'='*80}")
    print("CORRECTION PROGRESSION")
    print(f"{'='*80}")
    
    current_pitch = initial_pitch
    current_yaw = initial_yaw
    
    for pipe_num in range(1, 6):
        # Calculate correction (50% of remaining)
        pitch_correction = -current_pitch * 0.5
        yaw_correction = -current_yaw * 0.5
        
        # Calculate actual correction in mm
        pitch_correction_mm = pitch_correction * pipe_length_m
        yaw_correction_mm = yaw_correction * pipe_length_m
        
        # New values
        new_pitch = current_pitch + pitch_correction
        new_yaw = current_yaw + yaw_correction
        
        print(f"\nPipe {pipe_num}:")
        print(f"  Current:  Pitch={current_pitch:+7.2f} mm/m, Yaw={current_yaw:+7.2f} mm/m")
        print(f"  Correct:  Pitch={pitch_correction:+7.2f} mm/m, Yaw={yaw_correction:+7.2f} mm/m")
        print(f"            (Pitch={pitch_correction_mm:+7.2f} mm, Yaw={yaw_correction_mm:+7.2f} mm over {pipe_length_m}m)")
        print(f"  New:      Pitch={new_pitch:+7.2f} mm/m, Yaw={new_yaw:+7.2f} mm/m")
        
        # Check if close to zero
        if abs(new_pitch) < 0.5 and abs(new_yaw) < 0.5:
            print(f"  ✅ Close to target!")
            break
        
        current_pitch = new_pitch
        current_yaw = new_yaw


def get_float_input(prompt: str, default: Optional[float] = None) -> float:
    """Get float input from user with optional default"""
    while True:
        try:
            if default is not None:
                value = input(f"{prompt} [{default}]: ").strip()
                return float(value) if value else default
            else:
                value = input(f"{prompt}: ").strip()
                if not value:
                    print("  Please enter a value.")
                    continue
                return float(value)
        except ValueError:
            print("  Invalid input. Please enter a number.")


def interactive_simulator():
    """
    Interactive simulator that asks for current readings and ground condition
    """

    print("\n" + "="*80)
    print("INTERACTIVE STEERING CORRECTION SIMULATOR")
    print("="*80)
    print("\nThis simulator will help you plan corrections based on your actual readings.")
    print("\nYou can either:")
    print("  1. Enter current cylinder readings (we'll calculate pitch/yaw)")
    print("  2. Enter current pitch/yaw directly")

    # Ask input method
    print("\n" + "-"*80)
    input_method = input("\nEnter '1' for cylinder readings or '2' for pitch/yaw [1]: ").strip()
    input_method = input_method if input_method else '1'

    # Setup calculator
    params = MachineParameters(
        num_cylinders=3,
        stroke=50.0,
        mounting_diameter=715.0,
        pipe_length=3000.0,
        vertical_angle=1.49
    )
    calc = SteeringCalculator(params)

    current_pitch = 0.0
    current_yaw = 0.0

    if input_method == '1':
        # Get cylinder readings
        print("\n" + "="*80)
        print("ENTER CURRENT CYLINDER READINGS")
        print("="*80)
        print("Enter the current positions of your steering cylinders (in mm):\n")

        cyl1 = get_float_input("Cylinder 1 (Top, 0°)", default=25.0)
        cyl2 = get_float_input("Cylinder 2 (120°)", default=25.0)
        cyl3 = get_float_input("Cylinder 3 (240°)", default=25.0)

        # Calculate pitch/yaw from cylinders
        readings = CylinderReadings(cylinder_1=cyl1, cylinder_2=cyl2, cylinder_3=cyl3)
        steering = calc.calculate_steering(readings)
        current_pitch = steering.pitch
        current_yaw = steering.yaw

        print(f"\nCalculated from cylinder readings:")
        print(f"  Pitch: {current_pitch:+.2f} mm/m")
        print(f"  Yaw:   {current_yaw:+.2f} mm/m")

    else:
        # Get pitch/yaw directly
        print("\n" + "="*80)
        print("ENTER CURRENT PITCH AND YAW")
        print("="*80)
        print("Enter your current steering state:\n")

        current_pitch = get_float_input("Current Pitch (mm/m)", default=0.0)
        current_yaw = get_float_input("Current Yaw (mm/m)", default=0.0)

    # Get ground condition
    print("\n" + "="*80)
    print("ENTER GROUND CONDITION")
    print("="*80)
    print("Ground condition affects maximum steering rates:")
    print("  • SOFT:  Maximum 10 mm/m (can handle aggressive steering)")
    print("  • MIXED: Maximum 4 mm/m (limit to prevent jacking pressure increase)")
    print("  • ROCK:  Maximum 2 mm/m (CRITICAL - exceeding can halt jacking!)")

    while True:
        ground_input = input("\nGround condition [soft/mixed/rock] [mixed]: ").strip().lower()
        if not ground_input:
            ground_input = "mixed"

        if ground_input in ['soft', 's']:
            ground_condition = GroundCondition.SOFT
            break
        elif ground_input in ['mixed', 'm']:
            ground_condition = GroundCondition.MIXED
            break
        elif ground_input in ['rock', 'r']:
            ground_condition = GroundCondition.ROCK
            break
        else:
            print("  Invalid input. Please enter: soft, mixed, or rock")

    max_rate = ground_condition.get_max_steering_rate()
    recommended_max = ground_condition.get_recommended_max()
    print(f"\nSelected: {ground_condition.value.upper()}")
    print(f"  Maximum allowed rate: {max_rate} mm/m")
    print(f"  Recommended max rate: {recommended_max} mm/m")

    # Get target pitch/yaw
    print("\n" + "="*80)
    print("ENTER TARGET PITCH AND YAW")
    print("="*80)
    print("Enter your target steering state (usually 0, 0 to be straight and level):\n")

    target_pitch = get_float_input("Target Pitch (mm/m)", default=0.0)
    target_yaw = get_float_input("Target Yaw (mm/m)", default=0.0)

    # Get correction strategy
    print("\n" + "="*80)
    print("CORRECTION STRATEGY")
    print("="*80)
    print("How aggressively do you want to correct?")
    print("  • Conservative: 30% per pipe (safer, takes longer)")
    print("  • Moderate:     50% per pipe (balanced)")
    print("  • Aggressive:   70% per pipe (faster, limited by ground condition)")

    strategy = input("\nStrategy [conservative/moderate/aggressive] [moderate]: ").strip().lower()
    if strategy == 'conservative' or strategy == 'c':
        correction_rate = 0.3
    elif strategy == 'aggressive' or strategy == 'a':
        correction_rate = 0.7
    else:
        correction_rate = 0.5  # moderate

    print(f"\nUsing {int(correction_rate*100)}% correction rate per pipe")

    # Run simulation
    print("\n\n")
    simulate_correction_to_zero(
        initial_pitch=current_pitch,
        initial_yaw=current_yaw,
        target_pitch=target_pitch,
        target_yaw=target_yaw,
        max_pipes=10,
        correction_rate=correction_rate,
        ground_condition=ground_condition
    )


def main():
    """
    Main demonstration function
    """

    print("\n" + "="*80)
    print("STEERING CORRECTION SIMULATOR")
    print("Understanding How to Bring Deviations to Zero")
    print("="*80)

    # Ask if user wants interactive or demo mode
    print("\nSelect mode:")
    print("  1. Interactive Mode - Use YOUR actual readings")
    print("  2. Demo Mode - See examples with different ground conditions")

    mode = input("\nEnter choice [1]: ").strip()

    if mode == '2':
        # Part 1: Show how formulas work
        demonstrate_formulas()

        # Part 2: Show correction timeline
        show_correction_timeline()

        # Part 3: Full simulation (with different ground conditions)
        print("\n\n")
        print("="*80)
        print("SIMULATION 1: Soft Ground (More Aggressive Corrections Allowed)")
        print("="*80)
        simulate_correction_to_zero(
            initial_pitch=12.5,
            initial_yaw=-18.3,
            target_pitch=0.0,
            target_yaw=0.0,
            max_pipes=5,
            correction_rate=0.5,
            ground_condition=GroundCondition.SOFT
        )

        print("\n\n")
        print("="*80)
        print("SIMULATION 2: Mixed Ground (Limited to 2-4 mm/m)")
        print("="*80)
        simulate_correction_to_zero(
            initial_pitch=12.5,
            initial_yaw=-18.3,
            target_pitch=0.0,
            target_yaw=0.0,
            max_pipes=5,
            correction_rate=0.5,
            ground_condition=GroundCondition.MIXED
        )

        print("\n\n")
        print("="*80)
        print("SIMULATION 3: Rock Ground (Maximum 2 mm/m - Very Sensitive)")
        print("="*80)
        simulate_correction_to_zero(
            initial_pitch=12.5,
            initial_yaw=-18.3,
            target_pitch=0.0,
            target_yaw=0.0,
            max_pipes=5,
            correction_rate=0.5,
            ground_condition=GroundCondition.ROCK
        )

        print("\n" + "="*80)
        print("KEY TAKEAWAYS")
        print("="*80)
        print("""
1. Corrections must be applied PROGRESSIVELY over multiple pipes
2. Each pipe correction = steering_rate × pipe_length
3. Formula: Cylinder = center + (pitch × radius × cos(θ)) + (yaw × radius × sin(θ))
4. Monitor after each pipe and adjust plan
5. Don't over-correct - gradual is better than aggressive
6. Target is always: Pitch = 0 mm/m, Yaw = 0 mm/m
    """)
    else:
        # Interactive mode
        interactive_simulator()


if __name__ == "__main__":
    main()

